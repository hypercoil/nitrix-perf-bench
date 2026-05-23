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
honest. A **resource-aware scheduler** (`schedule.py`) then serialises GPU
attempts under a per-device **lock** (clean timings + clock stability) while
**parallel CPU** attempts run on **disjoint pinned cores** (`--cpu-slots N`,
honest because slots don't contend). Still to come in P1: the metric/baseline
registries and multi-platform result accumulation (DESIGN §8 — the multi-GPU /
mixed CPU+GPU fan-out the scheduler is already built for).

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
  uv run nperf --platform jax-cuda12 \
  --out reports/semiring_matmul.jsonl --report reports/PERF_SEMIRING_MATMUL.md
```

`--platform` picks the worker env-group (`jax-cpu` default / `jax-cuda12`);
worker interpreter resolves as `NPERF_PYTHON_<PLATFORM>` → `NPERF_WORKER_PYTHON`
→ this interpreter. `--cpu-slots N` runs N CPU attempts in parallel on disjoint
pinned cores (timings reflect the slot's core budget; `1` = full machine);
`--gpu-settle S` holds the device lock S seconds between GPU attempts.
`--out`/`--report` default to `results/<case>.{jsonl,md}`; `--quick` runs the
representative point, `--point '<json>'` a single explicit one, `--in-process`
uses the P0 driver, `--render-from <jsonl>` re-renders from saved rows. Tests:
`JAX_PLATFORMS=cpu uv run pytest` (CPU-only; schema, fidelity, case build,
worker round-trip, scheduler invariants).
