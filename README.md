# nitrix-perf-bench

Performance benchmark suite for [`nitrix`](../nitrix) — rigorous, fair,
multi-outcome, multi-platform, with structured results as the source of truth.

- **Architecture:** [`DESIGN.md`](DESIGN.md)
- **Row schema + worker lifecycle (implementation contract):**
  [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md)

## Status: P0a

The L0 measurement core + the L4 result schema, validated end-to-end against a
**throwaway** case (`dense_matmul`) and a markdown renderer. Results produced
under P0a are **disposable** (`schema_version = 0`); the schema freezes
additive-only at P0b.

## Run (P0a)

```bash
# with the project env (uv)
uv run nperf --quick

# or against any jax-capable interpreter
PYTHONPATH=src JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m nperf.run --quick
```

Writes structured rows to `results/<case>.jsonl` and a rendered report to
`results/<case>.md`.
