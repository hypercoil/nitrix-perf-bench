# Optimisation feedback loop — using perf-bench while improving a nitrix op

> **Audience: the nitrix agent.** This is the concrete command playbook for
> using `nitrix-perf-bench` as a *tight feedback loop* while you optimise a
> primitive — typically one the suite has flagged as **lagging on GPU** or
> slow on CPU. It assumes the L4 capsule layout (the `/scratch/nperf` envs);
> adapt the interpreter paths if you run elsewhere.

The loop in one sentence: **pick a flagged op → measure the baseline (one case,
one platform, fast) → edit the primitive → re-measure → keep the change only if
the ratio improved *and* fidelity still passes → gate-check the full case before
you commit.**

Two facts make this fast and safe:

- **nitrix is installed *editable*** in the bench venv
  (`__editable__.nitrix.pth → /root/capsule/code/nitrix/src`). Editing
  `src/nitrix/.../foo.py` takes effect on the **next worker spawn** — no
  reinstall, no copy. The inner loop is just "edit, re-run the case".
- **Fidelity is a hard gate, separate from speed.** Every row is scored against
  an fp64 oracle (`rel_to_tol ≤ 1 ⟺ ✓`). A change that is faster but flips a
  row to `✗` is a *regression*, not a win — the renderer refuses its ratio.

---

## 0. One-time environment

```bash
# source the capsule's bench env (heavy CUDA caches off the root overlay,
# persistent JAX compile cache OFF so cold-compile numbers are honest)
set -a; . /scratch/nperf/env.sh; set +a
export NPERF_PYTHON_CUPY=/scratch/nperf/venv-cupy/bin/python   # GPU-ref worker
PY=/scratch/nperf/venv/bin/python                              # jax+nitrix (editable)
```

`PY` is the orchestrator/worker interpreter (jax-cuda + editable nitrix). The
cupy GPU-reference baseline runs in its own env via `NPERF_PYTHON_CUPY`
(DESIGN §7). `JAX_PLATFORMS` is set per attempt by the runner — don't set it
yourself for a multi-platform run.

---

## 1. Pick the target

The deficit report ranks what to work on:

```bash
$PY tools/coverage_report.py                 # regenerates reports/COVERAGE_DEFICIT.md
```

In `reports/coverage_deficit.json`, each op carries `nitrix_slower_on_gpu`,
`gpu_ref_ratio` (the GPU ref's time ÷ nitrix's time at the representative point;
**< 1 means nitrix is slower**), and `gpu_ref` (which baseline beat it). The
`lagging` list is the work queue. Map the op's `qualname`
(`nitrix.geometry.spatial_transform`) to its **case name** — the case file is
`src/nperf/cases/<case>.py` and its `CASE.op_qualname` is the qualname.

---

## 2. The inner loop (fast — single case, single platform)

Measure only the platform you're changing and only the baselines you care
about. `--quick` runs the representative param point only; `--baselines`
restricts to an allowlist (the rest become cheap `skipped` rows — no worker, no
compile paid).

```bash
# baseline: nitrix vs the one reference that beats it, on the GPU, rep point only
$PY -m nperf.run --case spatial_transform --quick --platforms jax-cuda12 \
    --baselines nitrix-jax,cupyx.scipy.ndimage.map_coordinates \
    --worker-timeout 180
```

Read the rendered table (`results/spatial_transform.md`): the columns that
matter are **`fidelity`** (must stay `✓`), **`ratio`** (`Nx vs nitrix-jax` —
your optimisation target), and the two **guardrails** `compile` and `mem` (see
below). Note the `steady (min/med)` for nitrix-jax.

Now **edit the primitive** (`/root/capsule/code/nitrix/src/nitrix/...`) and
re-run the *exact same command*. The worker re-imports the edited nitrix. Compare
the new nitrix-jax `steady` and the `ratio`. Iterate.

- Working a **CPU** path (e.g. the B17 median cliff)? Use `--platforms jax-cpu`.
- Want a non-representative size? `--point '{"n": 4096, "seed": 0}'` instead of
  `--quick`.
- Keep `--worker-timeout` set: a wedged GPU or an accidental `O(huge)` compile
  is then bounded to a failure row, not a 50-minute hang.

**Correctness first.** If your faster variant changes the convention (a
different boundary, a dropped term), fidelity will flip to `✗`. That is the
signal to fix correctness before chasing speed — never commit a `✗`.

---

## 2a. Guardrail metrics — speed is not the only axis

`steady_time` is the headline, but a "win" that wrecks another metric is not a
win. Every row also reports two guardrails; watch them on the same table:

- **`compile` (compile_time)** — the *cold* first-call cost (`jax.clear_caches()`
  per attempt; persistent cache disabled, so this is honest). Aggressive fusion,
  loop unrolling (e.g. a larger `n_steps`), or `vmap`-over-everything can cut
  steady time while **inflating** compile from ~0.3 s to tens of seconds. The
  user pays compile once per shape, so a modest rise is fine — but a 10×
  blow-up for a 1.2× steady gain is usually the wrong trade. Flag it explicitly.
- **`mem` (peak_hbm on GPU / host_rss on CPU)** — per-attempt high-water (the
  subprocess runner makes it honest: the worker's peak *is* the attempt's peak).
  The classic speed-for-memory trade is materialising an intermediate that was
  previously streamed — e.g. building the dense `(n, n)` instead of a fused
  reduction. That can win steady time at small `n` and then **OOM at scale**.
  If `mem` climbs super-linearly with the size ladder, the change doesn't scale
  even if the rep-point ratio looks good.

Rule of thumb for accepting a change: **fidelity `✓`, steady ratio improved,
and neither `compile` nor `mem` materially regressed** (or the regression is a
deliberate, documented trade). When in doubt, report all four
(fidelity / steady / compile / mem) for before *and* after rather than just the
speedup — the tradeoff is the decision, not the headline number alone.

---

## 3. Rigorous before/after — gate mode

Eyeballing two tables misses regressions at the *other* param points. Capture a
baseline store **before** your change, capture the candidate **after**, and let
`gate.py` diff them (it exits non-zero if any key regressed on `steady_time`
min/p95):

```bash
# (a) BEFORE the edit — full case, both platforms, into a throwaway baseline dir
$PY -m nperf.run --case spatial_transform --platforms jax-cpu,jax-cuda12 \
    --worker-timeout 180 --store --out /tmp/before

# ... make the nitrix change ...

# (b) AFTER — measure again, then diff against the baseline
$PY -m nperf.run --case spatial_transform --platforms jax-cpu,jax-cuda12 \
    --worker-timeout 180 --store --out /tmp/after
$PY -m nperf.run --gate-baseline /tmp/before --gate-current /tmp/after \
    --gate-min 0.95 --gate-p95 0.95 --gate-out /tmp/gate.md
```

The gate writes a markdown diff and returns non-zero on regression — wire it as
the accept/reject check for the change. A real win shows the target op's ratio
improving with **no** other key regressing and **no** fidelity flip.

Note the gate keys on **`steady_time`** (min + p95). The guardrails — `compile`
and `mem` — are **not** auto-gated, so eyeball them in the before/after rendered
tables (or the store rows): confirm compile time and peak memory didn't balloon
as a hidden cost of the speedup.

---

## 4. Validate before committing

Once the change is good, run the full matrix into the durable store, re-render
the canonical report, and regenerate coverage so the lagging flag clears (or
doesn't):

```bash
$PY -m nperf.run --case spatial_transform --platforms jax-cpu,jax-cuda12 \
    --store --worker-timeout 180
$PY -m nperf.run --render-from results/store/spatial_transform --latest \
    --report reports/PERF_SPATIAL_TRANSFORM.md
$PY tools/coverage_report.py
JAX_PLATFORMS=cpu $PY -m pytest tests/ -q       # fidelity/oracle tests stay green
```

---

## 5. Did your change move a case's assumptions? (drift check)

A perf win usually changes an op's *signature* (a new default, e.g.
`backend='scan'` → `'auto'`) or its *output* on the benched input (a metric
going exact, a fallback firing). When it does, the perf-bench case that
measured the old behaviour can stay green while quietly measuring the wrong
branch. The drift gate catches exactly this:

```bash
JAX_PLATFORMS=cpu $PY tools/drift_check.py        # signature + behaviour vs manifest
# ... it flags the ops whose signature or output digest moved ...
JAX_PLATFORMS=cpu $PY tools/drift_check.py --update   # re-bless AFTER the cases are updated
```

It is a **change detector** (nitrix vs its own committed past), not a
correctness verdict — a flag means "the case's assumption moved, go re-read the
case", not "nitrix is wrong". Run it after any op change; if it flags, update
the affected case (the branch it calls, its accuracy gate) *before* trusting a
new number, then `--update` to re-bless. The fast signature half also runs in
`tests/test_op_drift.py` (default suite); the behaviour half is this tool.

---

## Worked example — the B17 CPU median/percentile cliff

`robust_zscore_normalize` is 8–12× slower than numpy on CPU because
`jnp.median` lowers to a full sort
([`median-percentile-cpu-sort-cliff`](../nitrix/docs/feature-requests/median-percentile-cpu-sort-cliff.md)).
To try a fix (e.g. a scipy-backed `pure_callback` median on the CPU backend,
branching on `jax.default_backend()`):

```bash
set -a; . /scratch/nperf/env.sh; set +a
PY=/scratch/nperf/venv/bin/python

# 1. baseline (CPU only, the regressed platform; nitrix vs the numpy floor)
$PY -m nperf.run --case robust_zscore_normalize --platforms jax-cpu \
    --baselines nitrix-jax,numpy.robust_zscore --worker-timeout 180 --store --out /tmp/before
#   read results/robust_zscore_normalize.md: nitrix steady ~1.18s @ n=2048,
#   ratio of numpy.robust_zscore ~0.09x (numpy 11x faster) — the gap to close.

# 2. edit src/nitrix/numerics/normalize.py (_median_mad / the CPU branch)

# 3. re-measure + gate
$PY -m nperf.run --case robust_zscore_normalize --platforms jax-cpu \
    --baselines nitrix-jax,numpy.robust_zscore --worker-timeout 180 --store --out /tmp/after
$PY -m nperf.run --gate-baseline /tmp/before --gate-current /tmp/after --gate-out /tmp/gate.md
```

Accept the change iff: nitrix-jax `steady` dropped, `fidelity` stayed `✓`
(the median is unchanged numerically), the `compile` and `mem` guardrails held
(a `pure_callback` adds little compile and no device memory — but confirm), the
gate passed, **and** GPU didn't regress — re-run step 4 on `jax-cpu,jax-cuda12`
to confirm the GPU path (which was already fast) is untouched. B17 records that
the in-jax candidates (`jnp.partition`, `lax.top_k`) do *not* close this gap —
measure, don't assume.

---

## If the op has no case yet

Cases live in `src/nperf/cases/<name>.py` and are registered in
`src/nperf/measure.py` (import + `_validate_case(...)`). A case needs a runtime
**array input the output depends on** (pure-generation ops constant-fold under
`jit` and time nothing — see the deferred `identity_grid`), an fp64 oracle, and
at least one GPU reference (cupy) to earn a *strong-GPU-ref* classification. See
[`DESIGN.md`](DESIGN.md) §L2 and any existing case (e.g. `cases/corr.py`) as a
template; verify the reference computes the **same** operation in fp64 before
wiring it (match-the-right-target).
```
