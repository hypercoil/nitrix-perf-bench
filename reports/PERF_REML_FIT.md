# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 039cf43ec5f270f26aac62e08fd731eb1b40563e | bench: d7e2dfb8fe3d41bea83814e40eb261cbc8dfb0fc
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T05:00:25.517321+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| reml_fit | jax-cpu | V=1024,k=8,n=24 | `nitrix-jax` | ok | 150.62 ms / 155.65 ms | 1.290 s | 693 MB (rss) | ✓ 0.31×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cpu | V=1024,k=8,n=24 | `statsmodels.MixedLM` | ok | 11.268 s / 11.432 s | 12.844 s | 559 MB (rss) | ✓ 0.057×tol | 74.81x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=1024,k=8,n=24 | `nitrix-jax` | ok | 11.29 ms / 11.52 ms | 2.001 s | 135.03 MB (hbm) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=1024,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |
| reml_fit | jax-cpu | V=256,k=8,n=24 | `nitrix-jax` | ok | 49.98 ms / 82.08 ms | 1.752 s | 674 MB (rss) | ✓ 0.29×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cpu | V=256,k=8,n=24 | `statsmodels.MixedLM` | ok | 2.920 s / 2.925 s | 3.526 s | 558 MB (rss) | ✓ 0.055×tol | 58.41x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=256,k=8,n=24 | `nitrix-jax` | ok | 11.35 ms / 11.50 ms | 2.193 s | 151.20 MB (hbm) | ✓ 0.11×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=256,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |
| reml_fit | jax-cpu | V=64,k=8,n=24 | `nitrix-jax` | ok | 12.80 ms / 14.67 ms | 1.172 s | 673 MB (rss) | ✓ 0.16×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cpu | V=64,k=8,n=24 | `statsmodels.MixedLM` | ok | 703.84 ms / 723.15 ms | 2.069 s | 557 MB (rss) | ✓ 0.04×tol | 55.01x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=64,k=8,n=24 | `nitrix-jax` | ok | 11.30 ms / 11.64 ms | 3.041 s | 151.05 MB (hbm) | ✓ 0.065×tol | 1.00x vs nitrix-jax |
| reml_fit | jax-cuda12 | V=64,k=8,n=24 | `statsmodels.MixedLM` | skipped | — | — | — | — | — |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

