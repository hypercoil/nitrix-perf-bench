# nitrix-perf-bench

Performance benchmark suite for [`nitrix`](../nitrix) — rigorous, fair,
multi-outcome, multi-platform, with structured results as the source of truth.

- **Architecture:** [`DESIGN.md`](DESIGN.md)
- **Row schema + worker lifecycle (implementation contract):**
  [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md)

## Status: P0b

The L4 result schema is **frozen at `schema_version = 1`** (additive-only from
here; `SCHEMA_AND_LIFECYCLE.md`). The first **real** nitrix case ships —
`semiring_matmul` — comparing the JAX reference, the Pallas/Triton kernel, and
a naive materialise-then-reduce baseline against an fp64 oracle, across the
`real` / `log` / `tropical` / `euclidean` algebras. Its rendered report
supersedes the hand-built `nitrix/bench/PERF_SEMIRING_MATMUL.md`. (The
throwaway `dense_matmul` case from P0a remains, for core smoke tests.)

Published reports live in [`reports/`](reports/) (the rendered markdown **and**
the L4 rows it was generated from, so the report is reproducible from committed
data). Scratch runs go to `results/` (git-ignored).

## Run

```bash
# CPU smoke via the project env (uv). The Pallas baseline records a `skipped`
# row off-GPU; nitrix-jax and naive-dense run.
JAX_PLATFORMS=cpu uv run nperf --quick

# Full sweep on a CUDA host (where the Pallas kernel actually runs), against
# any jax[cuda]-capable interpreter that can import nitrix:
PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m nperf.run \
  --out reports/semiring_matmul.jsonl --report reports/PERF_SEMIRING_MATMUL.md
```

`--out` defaults to `results/<case>.jsonl` and `--report` to
`results/<case>.md`; `--quick` runs the representative param point only.
Tests: `JAX_PLATFORMS=cpu uv run pytest` (CPU-only; schema + fidelity + case
build).
