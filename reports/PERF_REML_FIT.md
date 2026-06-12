# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 2fc75a1a139c5056929768793aa466c5e125e027 | bench: 5caed56afbe00b6b53c04b3ef7660c27d00fb684
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-12T17:55:05.696327+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| reml_fit | jax-cuda12 | V=1024,k=8,n=24 | `nitrix-jax` | ok | 11.29 ms / 11.52 ms | 2.001 s | 135.03 MB (hbm) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=1024,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |
| reml_fit | jax-cuda12 | V=256,k=8,n=24 | `nitrix-jax` | ok | 11.35 ms / 11.50 ms | 2.193 s | 151.20 MB (hbm) | ✓ 0.11×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=256,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |
| reml_fit | jax-cuda12 | V=64,k=8,n=24 | `nitrix-jax` | ok | 11.30 ms / 11.64 ms | 3.041 s | 151.05 MB (hbm) | ✓ 0.065×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=64,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

