# nitrix-perf-bench — design

> Decision-of-record for the nitrix performance-benchmark suite.  This
> document defines the **architecture**; specific nitrix ops to benchmark
> are out of scope here (they enter as *cases*, see §4).
>
> Status: design accepted 2026-05-22.  Substrate and environment-manager
> decisions are recorded in §3 and §7.

## 0. Why a separate repo

The suite lives outside `nitrix` for two hard reasons:

1. **Dependency isolation.** Fair benchmarking needs heavyweight reference
   implementations — PyTorch, PyTorch-Geometric, cuDNN-backed paths, SciPy —
   that `nitrix` must never depend on (SPEC §5.2). A sibling repo can install
   them in dedicated, isolated environments without contaminating the
   substrate.
2. **The existing `nitrix/bench/` design is insufficient.** It is a pile of
   bespoke scripts; the limits are architectural:
   - No framework — each `perf_*.py` re-implements `run`/`render`/`main`;
     adding a benchmark copies ~200 lines. `_util.py` abstracts only the
     stopwatch.
   - **One** hard-coded baseline per script; no way to host competing
     references.
   - Wall-time-centric; memory is a *separate* bespoke script
     (`mem_streaming_kernel.py`); no unified multi-metric notion.
   - One implicit platform (`jax.devices()[0]`), coupled to `nitrix`'s own
     venv.
   - Artefacts are hand-built markdown strings — not machine-readable, no
     provenance, no history, no regression detection.
   - Rigor stops at the median.

We **harvest** the prior art's good measurement instincts (warm-up / steady
median / `block_until_ready`; `memory_stats`; the readable `PERF_*.md` shape)
and rebuild the *design* around them.

## 1. Principles (the criteria, operationalised)

- **Structured results are the source of truth.** Every report, dashboard,
  and feed is *rendered* from the results datastore; nothing is hand-edited.
  → *interpretable, reproducible.*
- **Provenance on every datum.** nitrix SHA (+ dirty flag), bench SHA,
  resolved env-lock hash, device, jax version + backend, precision policy,
  config hash, timestamp, sample statistics. → *reproducible.*
- **Isolation by construction.** Each `(framework × platform)` runs in its own
  environment *and its own subprocess*. nitrix-jax and a torch baseline for
  the same op never share a process or a GPU context. → *fair, multi-platform.*
- **Measure, don't guess; co-report fidelity.** A wall-time ratio is emitted
  only alongside a numerical-agreement check against the reference, so we never
  compare apples to oranges. → *fair, rigorous.*
- **Curated and decision-driven.** This is not a microbench of every function;
  it benchmarks what informs a nitrix decision (the "benchmark-first" pattern —
  `G0_ELL_REPORT`, the trilinear baseline — made systematic). → *rigorous.*
- **Performance only.** Correctness is `nitrix`'s suite. This repo measures
  performance and the *fidelity of its own comparisons*, nothing else.

## 2. Layered architecture

```
L7  Orchestration / CLI         run · render · compare · gate
L6  Environments (uv matrix)     jax-cpu · jax-cuda12 · refs-torch · refs-pyg   (isolated, locked)
L5  Renderers                    markdown · HTML /site · op_matrix feed · regression diff · decision verdicts
L4  Results datastore            schema'd JSONL/Parquet, append-only, provenance-stamped   ← SOURCE OF TRUTH
L3  Runner                       sweep {case × param × baseline × platform}; one subprocess worker per env
L2  Cases + Baselines registry   declarative op descriptions + competing implementations
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
- **Compile vs steady-state.** Warm-up excludes first-call trace/compile;
  compile time is its own metric, not hidden.
- **Statistics.** Report the *distribution* — min (best achievable), median
  (typical), IQR / p95, n — not a point; auto-tune repeat count to a target
  relative CI; explicit outlier policy.
- **Honesty controls.** Inputs pre-placed on device with fixed seeds (exclude
  H2D); stable jit signatures (no recompiles in the timed region);
  **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** so `memory_stats` is truthful;
  GPU warm-up to steady clocks.

### L1 — Metrics (multi-outcome)
A metric is `(name, unit, direction, collect-protocol)`. Built-ins:
`steady_time` (min + median), `compile_time`, `peak_hbm` (`memory_stats`),
`host_rss`, `throughput` (elem/s), `est_flops` + `arithmetic_intensity`
(roofline), `fidelity_vs_ref` (max abs/rel error — the fairness guard), and
`energy` (NVML power × time) where available. Metrics are independent and
composable; a case declares which it supports.

### L2 — Cases + Baselines (extensible + fair)
- A **Case** declaratively describes one operation under test:
  `build(param_point) -> (setup, run, validate)`, the param space it spans
  (shape / dtype / algebra / …), and the metrics + baselines it supports.
  Adding an op = add one case module.
- A **Baseline** is a registered competing implementation of the *same math*:
  `nitrix-jax`, `nitrix-pallas`, `jnp-native`, `lax-native`, `scipy`, `torch`,
  `torch_geometric`, `cudnn`, a `naive` reference. **Multiple per op-family**,
  in a registry — directly fixing the single-baseline limitation. Each baseline
  declares its required environment and a *fidelity adapter*, so the harness can
  flag or refuse an unfair comparison.
- **Fairness contract** (harness-enforced): identical inputs (shared seed),
  matched precision policy (recorded — JAX `default_matmul_precision`, torch
  `allow_tf32`), same device, same I/O policy, both warmed + synced; the
  fidelity metric gates whether a ratio is "valid".

### L3 — Runner (multi-platform + isolation)
Config-driven sweep over `{case × param × baseline × platform}`. **One
subprocess worker per `(framework, platform)`**, dispatched via
`uv run --group <env>`, each emitting result records merged centrally — this
avoids torch/jax-cuda GPU contention and import clashes and yields clean
per-process `memory_stats`. Modes: `full`, `quick` (smoke), `targeted` (one
op), `compare` (two nitrix SHAs).

### L4 — Results datastore + provenance (the source of truth)
Append-only JSONL/Parquet, one row per `(case, param, baseline, platform,
metric)` carrying the full distribution **and** complete provenance (§1).
Versioned schema; runs are grouped by a run-id. Everything downstream is
derived from this — and its shape maps directly onto what
`nitrix/tools/op_matrix.py` already consumes.

### L5 — Renderers (artefacts)
All derived from L4:
- **Per-op markdown** — regenerate the `PERF_*.md` look, from data.
- **HTML `/site`** — interactive tables + plots (time-vs-size, roofline,
  history-over-SHA, platform comparison). (`.gitignore` already ignores
  `/site`.)
- **op_matrix feed** — emit the `{op → (cpu_baseline, cpu_ratio, gpu_baseline,
  gpu_ratio)}` JSON that `nitrix/tools/op_matrix.py` consumes, closing the loop
  into nitrix docs and filling today's `?` cells.
- **Regression diff** — current run vs a stored baseline run → deltas +
  thresholds, machine-readable for CI gating.
- **Decision verdicts** — the `G0_ELL_REPORT` / trilinear "benchmark-first"
  pattern as a template: given an op, its baselines, and a threshold, emit a
  recommendation (e.g. "JAX-default" / "pursue Pallas") that feeds
  BACKLOG / SPEC.

### L6 — Environments (uv matrix; see §7)
Isolated, locked envs as uv **dependency-groups**: `jax-cpu` (core),
`jax-cuda12` (pinned to nitrix's `jax[cuda12]==0.10.0` — replicating the
working A10G `/opt/jax_env`), `refs-torch`, `refs-pyg`. The conflicting
groups (`jax-cpu` vs `jax-cuda12`) are declared as uv conflicts. `nitrix`
enters as an **editable path source pinned by SHA**; that SHA lands in
provenance.

### L7 — Orchestration / CLI
`nperf run|render|compare|gate` (nox/just entrypoints under the hood), each
operating on configs (§5) and the datastore (L4).

## 3. Substrate decision

**Custom core + typed/Hydra config sweep** (own L0–L4): the rigorous
async-correct measurement, fairness, multi-framework subprocess isolation, and
results schema that off-the-shelf tools do not provide. Rationale: the
alternatives fight nitrix's hard requirements —

- **asv (airspeed velocity)** gives history / dashboard / regression / env
  matrix for free, but GPU-async timing, multi-framework fairness, and rich
  fidelity metrics must be shoehorned into custom `track_*` benchmarks, and its
  env matrix + process isolation are weak for our case.
- **pytest-benchmark** is single-metric (time), single-process, GPU-async-naive,
  no env matrix — useful only as a CPU micro-smoke.

L4 is designed so **asv can be layered downstream later** (consume our results
for longitudinal cross-SHA dashboards) without re-architecting. We do not start
on asv.

## 4. Feedback loop into nitrix

- **op_matrix feed** fills the existing `perf_ratio` columns (`< 1` = nitrix
  win) from real measurements.
- **Regression gate** (optional) on nitrix PRs touching perf-sensitive ops.
- **Decision verdicts** → BACKLOG / SPEC, systematising "benchmark-first".
- Results are attributed to a nitrix SHA, with a `compare`-two-SHAs mode.

## 5. Repo layout (concrete)

```
nitrix-perf-bench/
  pyproject.toml            # uv project; [dependency-groups] = the env matrix; nitrix path-source
  DESIGN.md                 # this document
  README.md
  noxfile.py                # entrypoints: run · render · compare · gate
  src/nperf/
    core/                   # L0–L1: timer, sync, memory, stats, provenance, metrics registry
    cases/                  # L2: one module per op-family (declarative); _base.py defines the protocol
    baselines/              # L2: competing-impl registry; one subpackage per framework
      torch/[pixi.toml?]    # per-baseline pixi escape hatch ONLY if a dep won't build via PyPI (§7)
    runner/                 # L3: sweep engine, subprocess workers, config schema
    report/                 # L5: markdown · html · op_matrix feed · regression · verdicts
    platforms/              # device/env descriptors + selection
  configs/                  # L7: typed/Hydra configs — suites, sweeps, platform matrices, thresholds
  results/                  # L4: structured outputs + schema (storage policy: see Open questions)
  site/                     # rendered HTML (gitignored)
```

## 6. Phasing (each phase ships a usable artefact)

- **P0** — L0 core + L4 schema + one case + markdown renderer → replace one
  `PERF_*.md` from data.
- **P1** — metric registry (time + peak-HBM + fidelity) + baseline registry +
  `jax-cpu` / `jax-cuda12` envs.
- **P2** — multi-framework refs (torch / PyG) in isolated envs + op_matrix feed.
- **P3** — HTML `/site` + regression gate + decision verdicts.

## 7. Environment-manager decision

**uv only**, to start — consistency with `nitrix` / `thrux`. The
platform × framework matrix is expressed as uv **dependency-groups** (with
`jax-cpu` ⟂ `jax-cuda12` declared as conflicting); the runner activates the
right group per subprocess worker via `uv run --group <env>`. `nitrix` is a
path source pinned by SHA.

**Escape hatch:** if — and only if — a legacy baseline's C++/CUDA dependency
cannot be compiled / installed via PyPI, drop a `pixi.toml` into *that specific
baseline's* reference folder (`src/nperf/baselines/<fw>/`) to isolate the
headache there, without adopting pixi suite-wide.

## 8. Open questions (not blocking the architecture)

- **Results storage.** In-repo `results/` vs a dedicated results branch vs
  external store; history depth / retention.
- **Must-have v1 metrics.** Proposed floor: `steady_time`, `peak_hbm`,
  `fidelity_vs_ref`. Defer `energy`, `est_flops`/roofline.
- **CI-gate scope.** Which ops are gated, and the regression threshold /
  noise envelope.
