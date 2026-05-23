# nitrix-perf-bench

Performance benchmark suite for [`nitrix`](../nitrix) — rigorous, fair,
multi-outcome, multi-platform, with structured results as the source of truth.

- **Architecture:** [`DESIGN.md`](DESIGN.md)
- **Row schema + worker lifecycle (implementation contract):**
  [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md)

## Status: P1 (subprocess runner)

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
per device at a time, N concurrent), each pinned via `CUDA_VISIBLE_DEVICES`. The
last P1 piece is a durable multi-device results store (DESIGN §8).

Published reports live in [`reports/`](reports/) (the rendered markdown **and**
the L4 rows it was generated from, so the report is reproducible from committed
data). Scratch runs go to `results/` (git-ignored).

## Run

The default runner spawns **one subprocess per attempt** (P1) so per-attempt
memory and cold-compile are honest. `--in-process` keeps the faster P0 driver
(memory metrics become process high-water marks — the report says so).

```bash
# CPU smoke (subprocess workers reuse this uv interpreter). The Pallas baseline
# records a `skipped` row off-GPU; nitrix-jax and naive-dense run.
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

# Accumulate separate runs/devices into one multi-platform report:
uv run nperf --render-from results/a10g.jsonl results/l40.jsonl \
  --report reports/combined.md
```

`--platforms` is a comma-list of worker env-groups (`jax-cpu` / `jax-cuda12`);
attempts fan out across them and distinct resources run in parallel. Worker
interpreter resolves as `NPERF_PYTHON_<PLATFORM>` → `NPERF_WORKER_PYTHON` → this
interpreter. `--cpu-slots N` runs N CPU attempts in parallel on disjoint pinned
cores (timings reflect the slot's core budget; `1` = full machine); `--gpus N`
fans GPU attempts across N devices (default: auto-probed), one lock each;
`--gpu-settle S` holds a device's lock S seconds between its attempts.
`--out`/`--report` default to `results/<case>.{jsonl,md}`; `--quick` runs the
representative point, `--point '<json>'` a single explicit one, `--in-process`
uses the P0 driver, `--render-from f1 [f2 …]` re-renders (and combines) saved
rows. Tests: `JAX_PLATFORMS=cpu uv run pytest` (CPU-only; schema, fidelity, case
build, worker round-trip, scheduler invariants, multi-platform).
