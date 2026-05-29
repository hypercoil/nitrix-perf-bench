# nitrix-perf-bench

Performance benchmark suite for [`nitrix`](../nitrix) — rigorous, fair,
multi-outcome, multi-platform, with structured results as the source of truth.

- **Architecture:** [`DESIGN.md`](DESIGN.md)
- **Row schema + worker lifecycle (implementation contract):**
  [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md)

## Status: P3 done (gate · bundles · HTML site)

The L4 result schema is **frozen at `schema_version = 1`** (additive-only;
`SCHEMA_AND_LIFECYCLE.md`). The first **real** nitrix case ships —
`semiring_matmul` — comparing the JAX reference, the Pallas/Triton kernel, and
a naive materialise-then-reduce baseline against an fp64 oracle, across the
`real` / `log` / `tropical` / `euclidean` algebras. Its rendered report
supersedes the hand-built `nitrix/bench/PERF_SEMIRING_MATMUL.md`. (The
throwaway `dense_matmul` case from P0a remains, for core smoke tests.)

**P1 (in progress):** the runner spawns **one subprocess per attempt** via a
pluggable interpreter — making per-attempt `peak_hbm` and cold `compile_time`
honest. A **resource-aware scheduler** (`schedule.py`) serialises GPU attempts
under a per-device **lock** (clean timings + clock stability) while **parallel
CPU** attempts run on **disjoint pinned cores** (`--cpu-slots N`, honest because
slots don't contend). **Multi-platform** works end to end: `--platforms a,b`
fans attempts across platforms in one run (CPU + a GPU overlap), and
`--render-from f1 f2 …` combines separate runs/devices into one `platform`-column
report with within-platform ratios. **Registries** are in: a metric registry
(units / direction / kind + the fidelity gate threshold, the single source of
truth the driver stamps and cases are validated against) and a **provider**
registry (the cross-case framework + env-isolation a baseline runs under — the
"baseline registry", realised on providers because baseline *names* are
case-local and would collide). **Multi-GPU fan-out** is in: `--gpus N` (default
auto-probed) gives each device its own lock, so attempts fan across GPUs (one
per device at a time, N concurrent), each pinned via `CUDA_VISIBLE_DEVICES`. A
**durable store** (`--store`; `results/store/<case>/<run_id>.jsonl`) accumulates
each run as a file, `--render-from <dir> --latest` combines runs/devices into
the current-state report, and `--prune-keep N` caps history. **P1 is complete**
modulo the cross-machine store transport policy (DESIGN §8).

**P2 (in progress) — cross-framework refs.** A `torch-dense` baseline (the same
materialise-then-reduce a torch practitioner writes for a non-real semiring
matmul) now runs as a `torch` **provider**: a separate, uv-isolated interpreter
(torch CPU wheels are on the PyTorch index, so it is its own *interpreter*, not
a second package manager). Build it reproducibly with
`tools/setup_refs_env.sh`, then point the runner at it with
`NPERF_PYTHON_TORCH`; the worker interpreter is now resolved per **(framework,
platform)**, so a torch attempt picks its own env even on a jax platform. With
no refs env configured, `torch-dense` records a clean `env_failed` row and the
jax baselines still run. The committed report combines the A10G GPU run with a
CPU cross-framework run.

A **PyG** baseline lands on a second case, `ell_edge_aggregate`, where it is the
*natural* reference: nitrix's `semiring_ell_edge_aggregate` is message passing
(gather ELL neighbours → per-edge `edge_fn` → semiring reduce) — exactly PyG's
`message`/`aggregate`. A torch `MessagePassing` baseline (GCN-style linear
`edge_fn`, so JAX / torch / the fp64 oracle compute identical math) competes
against `nitrix-jax` for sum- and max-aggregation (combined CPU+A10G report
`reports/PERF_ELL_EDGE_AGGREGATE.md`). The finding shows why measuring on the
target matters: PyG is ~2–5× faster on **CPU**, but on the **A10G** the gap
vanishes (nitrix within ~5% for sum, ~15% faster for max) — XLA fuses the
gather+vmap+reduce well on Ampere. Modern PyG message-passes on torch-native
`scatter_reduce`, so it installs pure-Python via uv (the same refs env) —
**not** the pixi escape hatch; pixi stays reserved for genuinely conda-only
compiled extensions. The GPU run uses a CUDA refs env
(`NPERF_REFS_VARIANT=cuda tools/setup_refs_env.sh`; jax[cuda12] + cuda torch
coexist, and torch's HBM is read from torch's own allocator).

The **op_matrix feed** (`tools/op_matrix_feed.py`) reads the accumulated rows
and emits the `perf_{cpu,gpu}_{baseline,ratio}` fields nitrix's
`docs/op_matrix.json` wants (ratio = nitrix.min / reference.min at the
representative point; `<1` = nitrix faster) — never mutating nitrix (`--apply`
writes a merged copy for review).

Published reports live in [`reports/`](reports/) (the rendered markdown **and**
the L4 rows it was generated from, so the report is reproducible from committed
data). Scratch runs go to `results/` (git-ignored).

## Run

The default runner spawns **one subprocess per attempt** (P1) so per-attempt
memory and cold-compile are honest. `--in-process` keeps the faster P0 driver
(memory metrics become process high-water marks — the report says so).

```bash
# CPU smoke (subprocess workers reuse this uv interpreter). The Pallas baseline
# records a `skipped` row off-GPU; nitrix-jax and naive-dense run; torch-dense
# records `env_failed` until the refs env below exists.
JAX_PLATFORMS=cpu uv run nperf --quick

# Cross-framework (P2): build the torch refs env once (off the root overlay --
# torch is ~1 GB), then point the runner at it. torch-dense now runs.
tools/setup_refs_env.sh                                  # -> $NPERF_REFS_ENV_DIR
NPERF_PYTHON_TORCH="${NPERF_REFS_ENV_DIR:-/output/nperf-refs-env}/bin/python" \
  JAX_PLATFORMS=cpu uv run nperf --quick

# Full sweep targeting a CUDA host. The orchestrator coordinates on CPU and
# spawns GPU workers via a pluggable interpreter; point it at a jax[cuda] env
# that can import nitrix:
NPERF_PYTHON_JAX_CUDA12=/path/to/cuda-env/bin/python \
  uv run nperf --platforms jax-cuda12 \
  --out reports/semiring_matmul.jsonl --report reports/PERF_SEMIRING_MATMUL.md

# Mixed run: CPU + GPU in one invocation (distinct resources run in parallel).
NPERF_PYTHON_JAX_CUDA12=/path/to/cuda-env/bin/python \
  uv run nperf --platforms jax-cpu,jax-cuda12

# Accumulate runs durably (one file per run), then render current state across
# every accumulated run/device:
NPERF_PYTHON_JAX_CUDA12=/path/to/cuda-env/bin/python \
  uv run nperf --platforms jax-cuda12 --store      # ingest A10G run
uv run nperf --platforms jax-cpu --store           # ingest a CPU run
uv run nperf --render-from results/store/semiring_matmul --latest \
  --report reports/combined.md                      # newest per (plat,param,baseline)

# Regression gate (P3): diff a current run against a stored baseline on
# steady_time min (tight) + p95 (loose); exits nonzero if either trips -> CI.
uv run nperf --gate-baseline reports/semiring_matmul.jsonl \
  --gate-current results/store/semiring_matmul \
  --gate-out results/gate.json --report results/gate.md

# Decision-input bundle (P3): structured evidence for a human call, no verdict.
python tools/decision_bundle.py --case semiring_matmul \
  --from results/store/semiring_matmul

# HTML site (P3): one self-contained page (tables + inline-SVG plots) from the
# whole store (or --render-from <rows>); /site is git-ignored.
uv run nperf --site site --render-from results/store
```

`--platforms` is a comma-list of worker env-groups (`jax-cpu` / `jax-cuda12`);
attempts fan out across them and distinct resources run in parallel. Worker
interpreter resolves as `NPERF_PYTHON_<PLATFORM>` → `NPERF_WORKER_PYTHON` → this
interpreter. `--cpu-slots N` runs N CPU attempts in parallel on disjoint pinned
cores (timings reflect the slot's core budget; `1` = full machine); `--gpus N`
fans GPU attempts across N devices (default: auto-probed), one lock each;
`--gpu-settle S` holds a device's lock S seconds between its attempts.
`--store [DIR]` ingests the run durably (default `results/store`); `--prune-keep
N` caps history. `--out`/`--report` default to `results/<case>.{jsonl,md}`;
`--quick` runs the representative point, `--point '<json>'` a single explicit
one, `--baselines A,B` / `--skip-baselines X,Y` select (allowlist) / drop
(denylist) baselines — a dropped baseline is **recorded as a `skipped` row**
(omission is data, DESIGN §1), so you can dodge a pathological cold compile
(e.g. `--skip-baselines naive-dense`) without a silent gap. `--skip-slow` drops
the case's *declared* `slow_baselines` for fast dev cycles and stamps the run
`coverage_mode=fast`, so the op_matrix feed + regression gate **refuse to treat
it as authoritative** — run the full sweep (omit `--skip-slow`) at sprint end
(`COVERAGE_MANDATE.md` §7). `--in-process` uses the P0 driver, `--render-from
<files/dirs> [--latest]` re-renders (and combines) saved rows. `--gate-baseline <files/dirs>` runs the
regression gate (`--gate-current`, default the store; `--gate-min`/`--gate-p95`
thresholds; `--gate-out` artifact) and exits nonzero on a regression; `--site
[DIR]` renders the self-contained HTML site. The op_matrix feed
(`tools/op_matrix_feed.py`), decision-input bundles
(`tools/decision_bundle.py`), and the **coverage-&-deficit report**
(`tools/coverage_report.py` — joins the op catalogue with the store and ranks
under-covered + on-target-lagging ops for the nitrix agent) are sibling L5
artifacts over the same rows. Tests:
`JAX_PLATFORMS=cpu uv run pytest` (CPU-only; schema, fidelity, case build,
worker round-trip, scheduler invariants, multi-platform, registries, store,
gate, bundle, html).
