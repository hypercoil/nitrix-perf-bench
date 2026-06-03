# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 645ce27d898f29997eff5632fb251170ec24d312 | bench: 126d3c00c62903f6e77922b34c777532d2a08b63
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T02:26:23.514131+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| polynomial_kernel | jax-cpu | n=2048,d=64 | `cupy.polynomial_kernel` | skipped | — | — | — | — | — |
| polynomial_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 3.56 ms / 5.67 ms | 70.99 ms | 864 MB (rss) | ✓ 0.0019×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cpu | n=2048,d=64 | `sklearn.polynomial_kernel` | ok | 57.14 ms / 62.76 ms | 66.41 ms | 864 MB (rss) | ✓ 0.0019×tol | 16.03x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=2048,d=64 | `cupy.polynomial_kernel` | ok | 290.3 µs / 294.7 µs | 176.24 ms | 0.52 MB (hbm) | ✓ 0.0019×tol | 1.72x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 169.1 µs / 170.2 µs | 433.11 ms | 105.38 MB (hbm) | ✓ 0.0019×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.polynomial_kernel` | ok | 58.88 ms / 90.40 ms | 98.58 ms | 0.52 MB (hbm) | ✓ 0.0019×tol | 348.13x vs nitrix-jax |
| polynomial_kernel | jax-cpu | n=4096,d=64 | `cupy.polynomial_kernel` | skipped | — | — | — | — | — |
| polynomial_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 30.12 ms / 50.92 ms | 186.94 ms | 897 MB (rss) | ✓ 0.0026×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cpu | n=4096,d=64 | `sklearn.polynomial_kernel` | ok | 203.04 ms / 210.50 ms | 256.93 ms | 865 MB (rss) | ✓ 0.0026×tol | 6.74x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=4096,d=64 | `cupy.polynomial_kernel` | ok | 2.01 ms / 2.02 ms | 178.27 ms | 1.05 MB (hbm) | ✓ 0.0026×tol | 2.13x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 940.1 µs / 946.2 µs | 272.59 ms | 206.57 MB (hbm) | ✓ 0.0026×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.polynomial_kernel` | ok | 194.62 ms / 195.39 ms | 191.86 ms | 1.05 MB (hbm) | ✓ 0.0026×tol | 207.02x vs nitrix-jax |
| polynomial_kernel | jax-cpu | n=512,d=64 | `cupy.polynomial_kernel` | skipped | — | — | — | — | — |
| polynomial_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 582.5 µs / 627.1 µs | 71.21 ms | 864 MB (rss) | ✓ 0.0019×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cpu | n=512,d=64 | `sklearn.polynomial_kernel` | ok | 3.50 ms / 3.67 ms | 3.61 ms | 864 MB (rss) | ✓ 0.0019×tol | 6.00x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=512,d=64 | `cupy.polynomial_kernel` | ok | 80.5 µs / 83.0 µs | 310.75 ms | 0.13 MB (hbm) | ✓ 0.0019×tol | 0.69x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 116.5 µs / 136.2 µs | 219.65 ms | 74.45 MB (hbm) | ✓ 0.0019×tol | 1.00x vs nitrix-jax |
| polynomial_kernel | jax-cuda12 | n=512,d=64 | `sklearn.polynomial_kernel` | ok | 3.30 ms / 10.19 ms | 11.07 ms | 0.13 MB (hbm) | ✓ 0.0019×tol | 28.35x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

