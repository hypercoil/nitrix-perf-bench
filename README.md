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

**P1 (in progress):** the runner now spawns **one subprocess per attempt** via
a pluggable interpreter — making per-attempt `peak_hbm` and cold `compile_time`
honest (each attempt's own process; the in-process driver's high-water-mark
caveat no longer applies by default). Still to come in P1: parallel scheduling
under the device lock, the metric/baseline registries, and multi-platform
result accumulation (DESIGN §8).

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
→ this interpreter. `--out`/`--report` default to `results/<case>.{jsonl,md}`;
`--quick` runs the representative point, `--point '<json>'` a single explicit
one, `--render-from <jsonl>` re-renders a report from saved rows. Tests:
`JAX_PLATFORMS=cpu uv run pytest` (CPU-only; schema + fidelity + case build +
worker round-trip).
