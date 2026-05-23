# nitrix-perf-bench — design

> Decision-of-record for the nitrix performance-benchmark suite.  This
> document defines the **architecture**; specific nitrix ops to benchmark
> are out of scope here (they enter as *cases*, see §2/L2).
>
> Status: design accepted 2026-05-22; **revised after design review rounds 1 &
> 2, 2026-05-22** (dispositions in §9).  Substrate and environment-manager
> decisions are recorded in §3 and §7.  The **L4 row schema and worker
> lifecycle** — the implementation contract that P0a iterates and P0b freezes —
> live in the companion **`SCHEMA_AND_LIFECYCLE.md`**.

## 0. Why a separate repo

The suite lives outside `nitrix` for two hard reasons:

1. **Dependency isolation.** Fair benchmarking needs heavyweight reference
   implementations — PyTorch, PyTorch-Geometric, cuDNN-backed paths, SciPy —
   that `nitrix` must never depend on (SPEC §5.2). A sibling repo can install
   them in dedicated, isolated environments without contaminating the
   substrate.
2. **The existing `nitrix/bench/` design is insufficient.** It is a pile of
   bespoke scripts; the limits are architectural: no framework (each
   `perf_*.py` re-implements `run`/`render`/`main`); **one** hard-coded
   baseline per script; wall-time-centric with memory as a *separate* bespoke
   script; one implicit platform coupled to `nitrix`'s own venv; artefacts that
   are hand-built markdown strings (no provenance, history, or regression
   detection); rigor that stops at the median.

We **harvest** the prior art's good measurement instincts (warm-up / steady
median / `block_until_ready`; `memory_stats`; the readable `PERF_*.md` shape)
and rebuild the *design* around them.

## 1. Principles (the criteria, operationalised)

- **Structured results are the source of truth, and renderers do no
  arithmetic.** Every report, dashboard, and feed is *rendered* from the
  results datastore; nothing is hand-edited. No metric is ever computed in a
  renderer — all arithmetic (ratios, errors, throughput, trends) lives in L1
  and is *stored*; renderers only select and format. → *interpretable,
  reproducible.*
- **Provenance on every datum** (full list in §1.1). → *reproducible.*
- **Isolation *and* non-contention.** Each `(framework × platform)` runs in its
  own environment and its own subprocess (isolation); and GPU-bound workers run
  serially per physical device under a device lock (non-contention) so they
  never corrupt each other's timings (see L3). → *fair, multi-platform.*
- **Measure, don't guess; co-report fidelity, and refuse-but-record.** A
  wall-time ratio is emitted *only* when the per-baseline fidelity adapter
  confirms the comparison is valid against a shared ground-truth oracle (L2).
  When it is not, the harness **refuses the ratio** but still records both
  absolute measurements *and* the fidelity-failure as a row with a reason — so
  an invalid comparison is *explained*, never silently dropped nor quietly
  mislabelled "valid". → *fair, rigorous.*
- **Two tiers of coverage, one rigor floor.** (a) A **coverage tier**: every
  shipping op benchmarked at one declared *representative param point*, at the
  full rigor floor — this is what fills `op_matrix`'s `?` cells. (b) A
  **decision tier**: deep parameter sweeps layered on top, only where a nitrix
  decision is live (the "benchmark-first" pattern — `G0_ELL_REPORT`, the
  trilinear baseline — made systematic). Same measurement core for both; the
  difference is breadth, not rigor. → *rigorous, and honest about coverage.*
- **Performance only.** Correctness is `nitrix`'s suite. This repo measures
  performance and the *fidelity of its own comparisons*, nothing else.

### 1.1 Provenance (stamped on every datum)

nitrix git SHA (+ dirty flag) and bench SHA; the **resolved dependency
versions in human-readable form** (`torch 2.4.1 → 2.5.0` is more actionable in
a regression than a hash diff) **plus** the env-lock hash; jax version +
backend and `XLA_FLAGS` in effect; the device (name, compute capability, memory,
driver) and its **clock-lock state** (or an explicit note that clocks were not
locked); CPU model + frequency-governor state; NUMA binding; precision policy
(JAX `default_matmul_precision`, torch `allow_tf32`); the
`XLA_PYTHON_CLIENT_PREALLOCATE` setting (**must be `false`** for honest
`memory_stats` — recording it next to the memory metric makes a forgotten flag
visible rather than silently corrupting a whole class of measurement); config
hash; timestamp; and the sample statistics (n, warm-up count, repeat policy).

## 2. Layered architecture

```
L7  Orchestration / CLI         run · render · compare · gate
L6  Environments (uv matrix)     jax-cpu · jax-cuda12 · refs-torch · refs-pyg   (isolated, locked)
L5  Renderers                    markdown · HTML /site · op_matrix feed · regression diff · decision-input bundles
L4  Results datastore            schema'd JSONL/Parquet, append-only, provenance-stamped   ← SOURCE OF TRUTH
L3  Runner                       sweep {case × param × baseline × platform}; subprocess worker per env; device-exclusive
L2  Cases + Baselines registry   declarative op descriptions + competing implementations + fidelity oracle
L1  Metrics registry             pluggable outcomes (time · memory · fidelity · throughput · …)
L0  Measurement core             rigorous, async-correct stopwatch + memory + stats + provenance
```

Each upper layer depends only on the ones below; the core (L0–L1) has no
knowledge of nitrix, cases, or rendering.

### L0 — Measurement core (rigor)
Framework-agnostic, the load-bearing piece:
- **Async correctness.** `block_until_ready` on every output leaf (`tree_map`);
  a per-framework `sync` hook (`torch.cuda.synchronize`, …) so non-JAX
  baselines are timed honestly.
- **Compile vs steady-state, with a declared cache policy.** Warm-up excludes
  first-call trace/compile; `compile_time` is its own metric. Because JAX
  compile cost is cache-dependent, the policy is: **persistent compilation
  cache disabled** (not isolated-to-a-temp-dir) with **`jax.clear_caches()`
  between in-process measurements**, so each attempt's `compile_time` is a
  genuine *cold* compile (the cost a nitrix user sees on first import). The
  cache state is recorded in provenance. (Worker reuse + cache discipline are
  pinned in `SCHEMA_AND_LIFECYCLE.md` §B/§D.)
- **Statistics.** Report the *distribution* — min (best achievable), median
  (typical), IQR / p95, n — not a point; auto-tune repeat count to a target
  relative CI; explicit outlier policy. The **regression gate diffs on `min`
  *and* `p95`** (min = noise-robust best-case; p95 = distribution-shape
  regressions — e.g. a slow-path fusion firing part of the time — that leave
  min untouched); the full distribution is for human-facing reports. See
  `SCHEMA_AND_LIFECYCLE.md` §F.
- **Honesty controls.** Inputs pre-placed on device with fixed seeds (exclude
  H2D); stable jit signatures (no recompiles in the timed region);
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`; GPU warm-up to steady clocks, with
  clock state recorded (L3 device-exclusivity keeps it stable across a run).

### L1 — Metrics (multi-outcome)
A metric is `(name, unit, direction, collect-protocol)`. Built-ins:
`steady_time` (min + median), `compile_time`, `peak_hbm` (`memory_stats`),
`host_rss`, `throughput` (elem/s), `est_flops` + `arithmetic_intensity`
(roofline), `fidelity_vs_ref`, and `energy` (deferred). Metrics are independent
and composable; a case declares which it supports. Two metrics carry protocol
caveats:
- **`fidelity_vs_ref` returns a *structured* record**, not a scalar:
  `{status: pass|fail|inconclusive, max_abs, max_rel, n_mismatched,
  layout_normalised, oracle}` (the `oracle` block records which rung of the
  ladder produced ground truth — `SCHEMA_AND_LIFECYCLE.md` §C). A single number
  would hide exactly the disagreements that matter; the gate reads the
  structure (L2).
- **`energy` needs a different protocol.** NVML power is sampled at ~10–50 ms,
  comparable to or longer than many ops, so `power × steady_median` is
  meaningless for millisecond ops. Energy is measured by looping the op for a
  *fixed wall-clock duration* and integrating power — a distinct protocol from
  the steady-state-median used for time. Deferred (§8) but flagged so it is not
  bolted onto the wrong protocol later.

### L2 — Cases + Baselines (extensible + fair)
- A **Case** declaratively describes one operation under test:
  `build(param_point) -> (setup, run, validate)`, the param space it spans, the
  metrics + baselines it supports, and its **representative param point** (used
  by the coverage tier, so coverage numbers are stable and meaningful).
- A **Baseline** is a registered competing implementation of the *same math*:
  `nitrix-jax`, `nitrix-pallas`, `jnp-native`, `lax-native`, `scipy`, `torch`,
  `torch_geometric`, `cudnn`, a `naive` reference. **Multiple per op-family.**
  Each baseline declares: its required environment; an `isolation:` marker
  (`uv` default, or `pixi` when it needs the §7 escape hatch, so the runner
  dispatches it correctly); and a **fidelity adapter**.
- **The fidelity oracle (computed once per param point, shared across all
  baselines).** Ground truth is the case's reference op in **fp64**, computed
  **outside the timed region** (cost irrelevant for *time*), and **every**
  baseline — nitrix included — is scored against *that identical* oracle, never
  against another (lossy) baseline. This resolves "which baseline is right when
  three disagree": none of them is. Cost is irrelevant for time but not for
  *feasibility*, so the oracle follows a **ladder** (rung recorded in
  `fidelity.oracle.kind`): (1) **fp64 full**; (2) **fp64 on a deterministic,
  stratified subsample** — *valid only for `output_independent` ops* (elementwise
  / gather / per-row); **coupled** ops (global reductions, solvers, FFTs) skip
  this rung because you cannot fp64 a slice of a coupled result, and below a
  per-case minimum-meaningful `n` the result is `fidelity.status =
  inconclusive`, not a forced pass/fail; (3) **designated baseline** when no
  fp64 path exists at all — the record is then labelled "agreement with X", not
  "error vs truth". Pinned in `SCHEMA_AND_LIFECYCLE.md` §C.
- **The fairness contract** (harness-enforced): identical inputs (shared seed);
  matched, recorded precision policy; same device; same I/O policy; both warmed
  + synced; the adapter normalises layout / index conventions (NCHW↔NHWC, PyG↔
  nitrix sparse indexing, etc.) before comparing and sets `layout_normalised`.
  A per-case fidelity threshold (with a global default) gates ratio emission;
  on failure the ratio is **refused but the absolutes + reason are recorded**.

### L3 — Runner (multi-platform, isolated, non-contending)
Config-driven sweep over `{case × param × baseline × platform}`.
- **Isolation:** one subprocess worker per `(framework, platform)`, dispatched
  via `uv run --group <env>` (or pixi for `isolation: pixi` baselines), each
  emitting result records merged centrally — avoiding torch/jax-cuda context
  clashes and yielding clean per-process `memory_stats`.
- **Non-contention:** GPU-bound workers acquire a **per-physical-device lock**
  and run **serially per device**; parallelism is allowed only across distinct
  devices or on CPU. Without this, concurrent GPU workers corrupt each other's
  timings and clock state.
- **Failure is data, not silence.** Any worker that cannot complete — env fails
  to resolve, OOM, compile error, fidelity gate fails — emits a **recorded
  skip/failure row** in L4 with a structured reason. Silent omission would
  corrupt historical comparisons.
- **Modes:** `full`, `quick` (smoke), `targeted` (one op), `compare` (two
  nitrix SHAs), `gate` (CI regression check against a stored baseline run).

### L4 — Results datastore + provenance (the source of truth)
Append-only JSONL/Parquet, **one record per measurement *attempt***
`(case, param, baseline, platform)` — with per-metric distributions nested
under the attempt and a `status` enum, so a fidelity-failed attempt still
carries its absolute measurements. (Granularity, the full status set, and the
`failure_detail` shapes are pinned in **`SCHEMA_AND_LIFECYCLE.md` §A** — that
annex is the row contract.) The **schema is versioned**: P0a's schema is
treated as **disposable** (results discarded), and from P1 onward changes are
**additive-only with a `schema_version` field**, because migrations on an
append-only store are painful. Everything downstream derives from this; its
shape maps directly onto what `nitrix/tools/op_matrix.py` consumes.

### L5 — Renderers (artefacts; no *metric* arithmetic)
All derived from L4, with all *metric* arithmetic already done and stored in L1
(ratios, errors, throughput, regression deltas, trends). Renderers may do
*pure presentation transforms* (histogram binning, log-axis scaling,
tile-level percentile aggregation for display). Rule of thumb: if the number
could appear in a regression gate or a decision-input bundle it is L1's; if it
only shapes a pixel it is the renderer's (`SCHEMA_AND_LIFECYCLE.md` §G).
- **Per-op markdown** — regenerate the `PERF_*.md` look, from data.
- **HTML `/site`** — interactive tables + plots (time-vs-size, roofline,
  history-over-SHA, platform comparison). (`.gitignore` already ignores
  `/site`.)
- **op_matrix feed** — emit the `{op → (cpu_baseline, cpu_ratio, gpu_baseline,
  gpu_ratio)}` JSON that `nitrix/tools/op_matrix.py` consumes (fed by the
  coverage tier, §4), closing the loop into nitrix docs and filling `?` cells.
- **Regression diff** — current run vs a stored baseline run, on `min` (tight)
  *and* `p95` (loose), with thresholds; machine-readable for the `gate` mode.
- **Decision-input bundles** — *not* auto-emitted recommendations. Each bundle
  packages the structured inputs a human needs to make a "benchmark-first"
  call (ratios, the fidelity structure, the threshold check, the historical
  trend) for an op/question. The recommendation ("JAX-default" / "pursue
  Pallas") stays a human-curated layer on top — until there is enough decision
  history to calibrate, the suite must not let "the benchmark said so" (a noisy
  median crossing a threshold) drive engineering.

### L6 / L7 — Environments & orchestration
See §7 (envs) and §6/§5 (CLI lives in `noxfile.py`: `run · render · compare ·
gate`).

## 3. Substrate decision

**Custom core + typed/Hydra config sweep** (own L0–L4): the rigorous
async-correct measurement, fairness, multi-framework subprocess isolation, and
results schema that off-the-shelf tools do not provide.

- **asv** gives history / dashboard / regression / env-matrix for free, but
  GPU-async timing, multi-framework fairness, and structured fidelity must be
  shoehorned into custom `track_*` benchmarks, and its env-matrix + process
  isolation are weak for our case. L4 is shaped so **asv can be layered
  downstream later** (consume our results for longitudinal dashboards) without
  re-architecting. We do not start on asv.
- **pytest-benchmark** is single-metric, single-process, GPU-async-naive —
  useful only as a CPU micro-smoke.

## 4. Feedback loop into nitrix

- The **coverage tier** feeds `op_matrix`'s `perf_ratio` cells (`< 1` = nitrix
  win) from real measurements — reconciling "curation" with "coverage" (§1).
  `op_matrix` is **always** fed by the representative coverage point; if a
  decision-tier sweep finds a regime where the ratio inverts, that lives in the
  decision-input bundle, not in `op_matrix` (so the matrix stays a stable,
  one-point-per-op summary).
- **Regression gate** (`gate` mode, on `min` + `p95`) — optional, on nitrix PRs
  that touch perf-sensitive ops.
- **Decision-input bundles** → read by a human → BACKLOG / SPEC.
- Results are attributed to a nitrix SHA, with a `compare`-two-SHAs mode.

## 5. Repo layout (concrete)

```
nitrix-perf-bench/
  pyproject.toml            # uv project; [dependency-groups] = the env matrix; nitrix path-source (SHA-pinned)
  DESIGN.md                 # this document
  README.md
  noxfile.py                # entrypoints: run · render · compare · gate
  src/nperf/
    core/                   # L0–L1: timer, sync, memory, stats, provenance, metrics registry
    cases/                  # L2: one module per op-family (declarative); _base.py defines the protocol
    baselines/              # L2: competing-impl registry; one subpackage per framework
      torch/                #   (a pixi.toml lives here ONLY via the §7 escape hatch; baseline marks isolation: pixi)
    runner/                 # L3: sweep engine, subprocess workers, device locks, config schema
    report/                 # L5: markdown · html · op_matrix feed · regression · decision-input bundles
    platforms/              # device/env descriptors + selection
  configs/                  # L7: typed/Hydra configs — suites, sweeps, platform matrices, thresholds
  results/                  # L4: structured outputs + versioned schema (storage policy: §8)
  site/                     # rendered HTML (gitignored)
```

## 6. Phasing (each phase ships a usable artefact)

- **P0a** — L0 core + L4 **schema**, validated against a *throwaway* case +
  renderer. The schema is the thing most regretted later; P0a exists to
  iterate it freely. **Results from P0a are disposable.**
- **P0b** — the first *real* case + the real markdown renderer that replaces
  one `PERF_*.md` from data. Schema is now frozen-additive (`schema_version`).
- **P1** — subprocess runner + device-lock/parallel-CPU scheduler +
  multi-platform (done); metric registry (`steady_time` + `peak_hbm` +
  `fidelity` floor, units/direction/threshold) + **provider** registry (done) +
  `jax-cpu` / `jax-cuda12` envs. *The "baseline registry" is realised as a
  **provider** registry:* baseline names (`nitrix-jax`, …) are case-local labels
  (disambiguated by the `case` field) and would collide across cases, so the
  registry instead holds the cross-case run providers (framework + env
  isolation) a baseline maps onto.  **Multi-GPU fan-out done** (`--gpus N`: one
  device lock each, attempts fan across devices, pinned via
  `CUDA_VISIBLE_DEVICES`).  **Durable accumulation store done** (`store.py`:
  per-run files, `--store` / `--render-from <dir> --latest` / `--prune-keep`).
  **P1 complete** modulo the cross-machine store *transport* policy (§8).
- **P2 (essentially done)** — multi-framework refs in isolated envs +
  op_matrix feed.  Both cross-framework providers are **uv-isolated separate
  interpreters** (not a separate package manager), selected per attempt by a
  **framework-aware interpreter resolution** (`NPERF_PYTHON_TORCH`), and built
  reproducibly by `tools/setup_refs_env.sh`; a missing refs env records a clean
  `env_failed` row rather than failing the sweep.
  - **torch** — a `torch-dense` baseline (the materialise-then-reduce a torch
    practitioner writes for a non-real semiring matmul) on the dense
    `semiring_matmul` case; combined CPU+A10G report
    (`reports/PERF_SEMIRING_MATMUL.md`).
  - **PyG** — on the `ell_edge_aggregate` case, where it is the *natural*
    reference (nitrix's `semiring_ell_edge_aggregate` is message passing:
    gather ELL neighbours → per-edge `edge_fn` → semiring reduce, exactly PyG's
    `message`/`aggregate`).  A `pyg` baseline (torch `MessagePassing`, a
    GCN-style linear `edge_fn` so JAX / torch / fp64-oracle compute identical
    math) competes against `nitrix-jax` for REAL (`aggr='add'`) and
    TROPICAL_MAX_PLUS (`aggr='max'`); report `reports/PERF_ELL_EDGE_AGGREGATE.md`
    (finding: PyG ~2–5× faster than the nitrix reference on CPU).  Modern PyG
    message-passes on torch-native `scatter_reduce`, so it installs pure-Python
    via uv — **not** the pixi escape hatch (forcing pixi would fabricate the
    need the `pixi_reason` guard prevents).  §7 pixi stays reserved for if a
    baseline ever needs the *compiled* `torch-scatter`/`torch-sparse`
    extensions with no portable PyPI wheel for the torch/CUDA pin.
  - **op_matrix feed** (`tools/op_matrix_feed.py`): reads the accumulated rows
    and emits, per op, the `perf_{cpu,gpu}_{baseline,ratio}` fields nitrix's
    `docs/op_matrix.json` wants (ratio = nitrix-primary.min / reference.min at
    the representative point; `<1` = nitrix faster).  It never mutates nitrix —
    `--apply` writes a merged *copy* for review, so the op_matrix change is
    nitrix's own commit.
- **P3** — HTML `/site` + regression `gate` + decision-input bundles.

## 7. Environment-manager decision

**uv only**, to start — consistency with `nitrix` / `thrux`. The
platform × framework matrix is expressed as uv **dependency-groups** (with
`jax-cpu` ⟂ `jax-cuda12` declared as conflicting); the runner activates the
right group per subprocess worker via `uv run --group <env>`. `nitrix` is a
path source pinned by SHA (recorded in provenance).

**Escape hatch:** if — and only if — a legacy baseline's C++/CUDA dependency
cannot be installed via PyPI, drop a `pixi.toml` into *that specific baseline's*
reference folder (`src/nperf/baselines/<fw>/`) **and** mark that baseline
`isolation: pixi, reason: "<why PyPI failed>"` in the registry — the recorded
reason leaves an audit trail and keeps pixi an *escape from* the established uv
default, not a starting choice for a merely-painful install. The marker keeps
the escape hatch from quietly becoming a second, undeclared dispatch system.
**The device lock (L3) is dispatcher-agnostic:** it lives in the runner above
both uv- and pixi-spawned workers (a uv `nitrix-jax` worker and a pixi
`torch_geometric` worker on the same GPU serialise against the *same* lock —
`SCHEMA_AND_LIFECYCLE.md` §E).

## 8. Open questions (not blocking the architecture)

- **Multi-platform producing/rendering + the accumulation store: DONE (P1).**
  Every row is keyed by `platform` and carries device provenance (annex §A), so
  multi-platform was always a schema property, not a new axis. P1 delivered:
  `--platforms a,b` (+ `--gpus N`) fans attempts across resources in one run;
  the **store** (`store.py`, default git-ignored `results/store/<case>/<run_id>
  .jsonl`) accumulates each run as a file (`--store`), `--render-from <dir>`
  globs it, `--latest` collapses to the current row per `(case, platform, param,
  baseline)`, and `--prune-keep N` caps history (the failure-row-volume guard).
  A second device's run (a Lovelace L40 beside the A10G) is just another file —
  validated by combining a CPU and an A10G run into one `platform`-column report.
  **Still open — the *transport* policy only:** where the store lives across
  *machines* (local + git-ignored, a results branch, or a network/object store);
  the on-disk accumulation / selection / retention mechanism is built and
  machine-local today.
- **Must-have v1 metrics.** Floor proposed: `steady_time`, `peak_hbm`,
  `fidelity_vs_ref`. Defer `energy` (distinct protocol, L1) and
  `est_flops`/roofline.
- **CI-gate scope.** Which ops are gated, and the `min`/`p95` regression
  thresholds / noise envelope.

## 9. Review dispositions (2026-05-22)

- **Accepted:** refuse-don't-flag on invalid fidelity; renderers do no
  arithmetic; structured `fidelity_vs_ref` + per-case threshold; cold-cache
  compile policy; expanded provenance (CPU/governor, GPU clock-lock, NUMA,
  `XLA_FLAGS`, resolved human-readable versions) with the preallocate note
  adjacent; the two-tier coverage/decision reconciliation; P0a/P0b split;
  decision-verdicts → decision-*inputs*; energy's distinct protocol;
  `isolation: pixi` registry marker; `gate` mode; env-failure → recorded skip
  row.
- **Refined / added:** fidelity oracle = **fp64 reference** (every baseline
  scored against truth, not a designated baseline), with designated-baseline as
  labelled fallback; **device-exclusive (serial-per-GPU) scheduling** added to
  L3 as architecture (a non-contention gap the prior draft missed); regression
  gate diffs on **`min`**; "refuse" still **records** the absolutes + reason;
  the P0 split's load-bearing rule is **schema disposable in P0a,
  additive-only thereafter**, not the split ceremony itself.

### Round 2 (2026-05-22) — row schema & lifecycle

The residual gaps clustered in the L4 row shape and worker-lifecycle
interaction surfaces; they are cashed out in the **`SCHEMA_AND_LIFECYCLE.md`**
annex rather than re-revising the architecture.

- **Accepted:** renderer "no-arithmetic" carve-out (metric arithmetic in L1;
  presentation transforms in renderers); per-attempt row with status-shaped
  `failure_detail` (`fidelity_failed` rows keep their metrics); `isolation:
  pixi` carries a recorded `reason`; device lock is dispatcher-agnostic;
  results-storage ↔ failure-row-volume coupling; §1.1 stays the provenance
  reference.
- **Refined / pushed:** subsample is **valid only for `output_independent`
  ops** — coupled ops skip that rung and may be `fidelity_inconclusive` (a
  third state); cold-compile is resolved as **`jax.clear_caches()` between
  in-process attempts** (worker reuse without leaking warm compiles), cache
  **disabled** not isolated; the shape-shift guard is **`p95`** (not median),
  gating on `min` *and* `p95`; the oracle is **computed once per param point
  and shared** across baselines (enforcing identical truth).
