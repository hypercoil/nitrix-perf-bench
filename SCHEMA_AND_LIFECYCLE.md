# nitrix-perf-bench — L4 row schema & worker lifecycle

> Annex to `DESIGN.md` (architecture).  This is the **implementation
> contract** for the result row and the worker process — the surface that
> P0a iterates and P0b freezes (additive-only thereafter).  Written after
> design review round 2 (2026-05-22).

The two questions this annex answers: *what exactly does an L4 row look like
across `ok` / failure / fidelity-fallback / subsample cases?*, and *how do the
cold-compile discipline (L0) and the device lock (L3) survive the
multi-measurement-worker and uv+pixi-coexistence surfaces the revision
introduced?*

---

## A. The L4 row: one record per **measurement attempt**

The unit of record is a **measurement attempt**, keyed by
`(run_id, case, param_point, baseline, platform)`.  *Not* one row per metric:
`status` is a property of the attempt, and a fidelity-failed attempt still
carries time/memory measurements, so metrics are nested under the attempt.
(This supersedes the "one row per … metric" phrasing in DESIGN §L4.)  A
tidy/long view (one row per metric) is *derived* by the analytics layer when a
query wants it; the canonical store is per-attempt.

```jsonc
{
  // ---- identity / keys ----
  "schema_version": 2,
  "run_id": "2026-05-22T18:03:11Z__a1b2c3d",   // groups one invocation
  "case": "semiring_matmul",                    // op-family case id
  "param_point": {                              // structured -> filterable
    "shape": [512, 256, 512], "dtype": "f32", "algebra": "log",
    "tier": "coverage" | "decision",            // which suite tier produced it
    "representative": true                       // the coverage anchor point?
  },
  "baseline": "nitrix-pallas",                  // the implementation under test
  "platform": "jax-cuda12",                     // env-group id (L6)
  "framework": "jax",

  // ---- status (drives which optional blocks are present) ----
  "status": "ok",
    // ok | env_failed | compile_error | oom | timeout
    //    | fidelity_failed | fidelity_inconclusive | skipped

  // ---- metrics (present iff status in {ok, fidelity_failed, fidelity_inconclusive}) ----
  // fidelity_* still measured time/mem before the comparison gated the ratio.
  "metrics": {
    "steady_time":  {"min": 2.41e-3, "median": 2.55e-3, "p95": 2.9e-3, "iqr": 1.1e-4, "n": 12, "unit": "s"},
    "compile_time": {"value": 0.83, "unit": "s", "cache": "cold"},
    "peak_hbm":     {"value": 5.5,  "unit": "MB"},
    "throughput":   {"value": 2.7e10, "unit": "elem/s"}
    // raw per-sample times optional (volume); summary always present.
  },

  // ---- fidelity (present iff a comparison was attempted) ----
  "fidelity": {
    "status": "pass",                            // pass | fail | inconclusive
    "max_abs": 3.0e-6, "max_rel": 1.2e-6, "n_mismatched": 0,
    "layout_normalised": true,
    "oracle": {
      "kind": "fp64_full",                       // fp64_full | fp64_subsample | designated_baseline
      "baseline": null,                          // set iff kind == designated_baseline
      "subsample": null                          // set iff kind == fp64_subsample (block C)
    },
    "threshold": {"max_rel": 1e-4, "scope": "per_case"}
  },

  // ---- ratio (present iff status == ok AND fidelity.status == pass) ----
  // Computed in L1 and STORED; never recomputed in a renderer.
  "ratio": {"vs": "nitrix-jax", "metric": "min", "value": 0.41},

  // ---- failure_detail (present iff status != ok); shape depends on status ----
  "failure_detail": null,

  // ---- provenance (DESIGN §1.1) ----
  "provenance": { /* see DESIGN §1.1 — flattened or nested */ }
}
```

### `failure_detail` shapes (by `status`)

| `status` | phase | `failure_detail` |
|---|---|---|
| `env_failed` | pre-measurement | `{phase: "resolve"\|"import", env_group, message}` |
| `compile_error` | pre-measurement | `{jit_signature, message}` |
| `oom` | mid-measurement | `{requested_bytes?, device_free_bytes?, message}` |
| `timeout` | mid-measurement | `{limit_s}` |
| `fidelity_failed` | post-measurement | `{fidelity: <the structured record above>}` (metrics present) |
| `fidelity_inconclusive` | post-measurement | `{reason: "subsample_too_small"\|"no_oracle"\|"coupled_op_no_fp64", fidelity?}` |
| `skipped` | by config | `{reason}` |

The key property: **`fidelity_failed` and `fidelity_inconclusive` rows still
carry the `metrics` block** — the absolutes were measured; only the *ratio* was
refused.  This is what makes "how often does `nitrix-pallas` fidelity-fail on
`f16` but pass on `f32`" answerable (filter `status`, group by
`param_point.dtype`) — and keeps refusal from dropping data (DESIGN §1).

---

## B. Worker lifecycle

A **worker** is one OS process per `(framework, platform, dispatcher)`,
spawned by the runner via `uv run --group <env>` (or pixi for an
`isolation: pixi` baseline, DESIGN §7).  A worker handles **many attempts
serially** — process spawn is amortised, not per-measurement.

**Per-attempt loop (inside a worker):**

1. Receive an attempt spec (case, param_point, baseline, metrics).
2. `jax.clear_caches()` (and equivalent for non-JAX frameworks) — see §D.
3. Build inputs on-device (fixed seed); warm up (excluded from `steady_time`);
   record `compile_time` from the first call.
4. Run the timed loop (L0 protocol); collect metric distributions.
5. Fidelity: compare against the **pre-computed oracle for this param_point**
   (§C); fill the `fidelity` block; decide ratio emission.
6. Emit exactly one L4 row; continue to the next attempt.

**Failure is caught per attempt, never fatal to the worker or the sweep.**
Each attempt is wrapped: any exception is classified into a `status` +
`failure_detail`, the row is emitted, and the loop proceeds.  OOM specifically:
catch, record `device_free_bytes`/`requested_bytes` if obtainable,
`clear_caches()`, continue.

---

## C. The fidelity oracle (computed once per param point)

The oracle for a `(case, param_point)` is computed **once** and **shared across
every baseline** for that point — so all baselines (nitrix included) are scored
against the *identical* truth, and fp64 is not recomputed per baseline.

**Oracle ladder** (each rung recorded in `fidelity.oracle.kind`):

1. **`fp64_full`** — the case's reference op in fp64, if it fits in memory and
   is tractable.  Default.
2. **`fp64_subsample`** — *only if the op is `output_independent`* (each output
   element depends on a bounded, identifiable input subset — true for
   elementwise / gather / per-row ops; **false** for global reductions,
   iterative solvers, FFTs, anything fully coupled).  For coupled ops this rung
   is **skipped** — you cannot fp64 a slice of a coupled result.  When used,
   record `oracle.subsample = {n, fraction, seed, stratification}`
   (deterministic, stratified across the output index space) and **refuse to
   elide that field**.  Below a per-case minimum-meaningful `n` →
   `fidelity.status = inconclusive`, not a forced pass/fail.
3. **`designated_baseline`** — when no fp64 path exists at all (integer/gather
   kernels; sizes where even subsampled fp64 is infeasible *and* the op is
   coupled).  `oracle.baseline` names it, and the fidelity number is explicitly
   "agreement with X", not "error vs truth".

`output_independent` is declared by the case (or its fidelity adapter); the
adapter also owns layout/index normalisation (NCHW↔NHWC, PyG↔nitrix indexing)
and sets `layout_normalised` before comparing.

---

## D. Compile-cache discipline (resolves the L0×lifecycle interaction)

The persistent on-disk JAX cache is **disabled** for the whole worker
(`JAX_COMPILATION_CACHE_DIR` unset — *disabled*, not isolated-to-a-temp-dir);
provenance records `compile_cache: "disabled"`.  Because one worker handles many
attempts in-process, **`jax.clear_caches()` runs between attempts** so each
attempt's first call is a genuine **cold** compile — the cost a nitrix user
sees on first import — and `metrics.compile_time.cache == "cold"` is honest.
The recompile that warm-up then triggers is excluded from `steady_time` by
construction, so this costs nothing in measurement validity.  (Within a single
attempt, the warm-up + timed runs share the in-process cache — that is correct;
`steady_time` is the *warm* steady state.)

---

## E. Device lock (resolves the L3 × uv/pixi-coexistence interaction)

The per-physical-device lock lives in the **runner/orchestrator, above both
dispatch paths** — *not* inside the uv worker.  The GPU does not care which
package manager spawned the process, so a uv-spawned `nitrix-jax` worker and a
pixi-spawned `torch_geometric` worker on the same A10G must serialise against
the *same* lock.  The runner acquires the device lock before handing GPU work
to a worker (regardless of dispatcher) and holds it for that attempt's GPU
phase.  CPU attempts and attempts on distinct physical devices proceed in
parallel.  GPU attempts on one device run back-to-back under the lock to keep
clock state stable; an optional settle interval is recorded in provenance.

---

## F. Regression gate (dual signal)

The `gate` mode diffs the current run against a stored baseline run on **two**
statistics, and fires if **either** trips:

- **`min`** — tight threshold; noise-robust detection of best-case slowdowns.
- **`p95`** — loose threshold; catches **distribution-shape** regressions (a
  slow-path XLA fusion that fires e.g. 30% of the time, or bimodality) that
  leave `min` untouched.

Both thresholds live in config and are stamped into the gate artifact.

---

## G. Renderer arithmetic carve-out

"Renderers do no arithmetic" (DESIGN §1) means **no *metric* arithmetic**:
ratios, errors, throughput, regression deltas, and trend slopes are computed in
L1 and **stored** in the row.  **Pure presentation transforms** — histogram
binning, log-axis scaling, tile-level percentile-of-percentile for a dashboard
cell — are renderer work on already-stored values.  Rule of thumb: *if the
number could appear in a regression gate or a decision-input bundle, it is L1;
if it only shapes a pixel, it is the renderer's.*
