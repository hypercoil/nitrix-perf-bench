# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 11f91e38159ffe403128b6cf08cf52be19798870 | bench: 14b8ae1fea1eae07a45ad92503d9604a2be294a4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T22:45:39.099495+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `nitrix-jax` | ok | 90.33 ms / 93.58 ms | 238.87 ms | 1440 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg` | ok | 221.53 ms / 261.27 ms | 1.543 s | 1440 MB (rss) | ✓ 0.02×tol | 2.45x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 222.78 ms / 279.61 ms | 1.675 s | 1440 MB (rss) | ✓ 0.021×tol | 2.47x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `nitrix-jax-poly` | ok | 312.59 ms / 337.49 ms | 1.952 s | 1440 MB (rss) | ✓ 0.51×tol | 3.46x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `nitrix-jax-shift_invert` | ok | 521.31 ms / 540.94 ms | 2.014 s | 1440 MB (rss) | ≈ 2.3×tol | 5.77x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=32,fmt=dense | `scipy.sparse.eigsh` | ok | 79.51 ms / 82.75 ms | 86.11 ms | 1440 MB (rss) | ✓ 3.8e-11×tol | 0.88x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `cupyx.sparse.eigsh` | ok | 399.18 ms / 891.53 ms | 2.026 s | 4.19 MB (hbm) | ✓ 0.00037×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg` | ok | 89.13 ms / 92.44 ms | 5.770 s | 138.68 MB (hbm) | ✓ 0.021×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 88.58 ms / 92.74 ms | 7.456 s | 138.68 MB (hbm) | ✓ 0.016×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-poly` | ok | 49.07 ms / 49.64 ms | 6.702 s | 138.68 MB (hbm) | ✓ 0.57×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-shift_invert` | ok | 51.18 ms / 53.11 ms | 6.909 s | 105.13 MB (hbm) | ≈ 2.2×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `scipy.sparse.eigsh` | ok | 378.35 ms / 775.82 ms | 778.09 ms | 4.19 MB (hbm) | ✓ 4.6e-11×tol | — |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax` | ok | 91.90 ms / 98.20 ms | 438.37 ms | 1440 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 44.90 ms / 46.81 ms | 1.295 s | 1440 MB (rss) | ✓ 0.018×tol | 0.49x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 53.76 ms / 58.82 ms | 1.423 s | 1440 MB (rss) | ✓ 0.014×tol | 0.58x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 63.48 ms / 65.21 ms | 1.497 s | 1440 MB (rss) | ≈ 2.3×tol | 0.69x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 125.62 ms / 127.24 ms | 1.738 s | 1440 MB (rss) | ≈ 12×tol | 1.37x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 51.89 ms / 53.43 ms | 53.70 ms | 1440 MB (rss) | ✓ 1.7e-11×tol | 0.56x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 41.35 ms / 62.43 ms | 1.330 s | 4.19 MB (hbm) | ✓ 0.00034×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 29.55 ms / 35.27 ms | 3.828 s | 138.49 MB (hbm) | ✓ 0.012×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 32.25 ms / 33.45 ms | 4.129 s | 138.49 MB (hbm) | ✓ 0.011×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 16.75 ms / 17.00 ms | 4.296 s | 138.49 MB (hbm) | ≈ 2.2×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 23.73 ms / 25.18 ms | 4.461 s | 155.37 MB (hbm) | ≈ 11×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 79.59 ms / 152.21 ms | 215.16 ms | 4.19 MB (hbm) | ✓ 1.3e-11×tol | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax` | ok | 527.67 ms / 542.18 ms | 666.90 ms | 1440 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 145.71 ms / 148.74 ms | 1.428 s | 1440 MB (rss) | ✓ 0.09×tol | 0.28x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 212.23 ms / 223.59 ms | 1.587 s | 1440 MB (rss) | ✓ 0.075×tol | 0.40x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 228.01 ms / 229.58 ms | 1.500 s | 1440 MB (rss) | ≈ 5.9×tol | 0.43x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 315.74 ms / 320.03 ms | 1.937 s | 1440 MB (rss) | ≈ 21×tol | 0.60x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 247.53 ms / 262.46 ms | 247.67 ms | 1440 MB (rss) | ✓ 3.6e-11×tol | 0.47x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 166.93 ms / 218.74 ms | 1.045 s | 16.78 MB (hbm) | ✓ 0.00032×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 33.17 ms / 35.79 ms | 6.008 s | 117.44 MB (hbm) | ✓ 0.05×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 33.95 ms / 35.68 ms | 6.310 s | 121.64 MB (hbm) | ✓ 0.047×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 19.65 ms / 20.24 ms | 4.616 s | 117.96 MB (hbm) | ≈ 4.3×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 20.99 ms / 23.26 ms | 5.333 s | 151.14 MB (hbm) | ≈ 21×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 1.961 s / 2.453 s | 2.051 s | 16.78 MB (hbm) | ✓ 2e-11×tol | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 198.85 ms / 201.03 ms | 1.467 s | 1440 MB (rss) | ✓ 0.09×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 210.22 ms / 210.80 ms | 1.644 s | 1440 MB (rss) | ✓ 0.075×tol | 1.06x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 342.73 ms / 348.75 ms | 1.959 s | 1440 MB (rss) | ≈ 5.9×tol | 1.72x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 489.31 ms / 494.21 ms | 2.149 s | 1440 MB (rss) | ≈ 21×tol | 2.46x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 239.34 ms / 257.02 ms | 257.58 ms | 1440 MB (rss) | ✓ 2.7e-11×tol | 1.20x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 63.27 ms / 131.18 ms | 851.62 ms | 20.15 MB (hbm) | ✓ 0.00035×tol | 0.63x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 101.12 ms / 134.53 ms | 5.291 s | 120.96 MB (hbm) | ✓ 0.047×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 85.97 ms / 87.94 ms | 4.144 s | 120.96 MB (hbm) | ✓ 0.049×tol | 0.85x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 129.54 ms / 186.57 ms | 3.769 s | 120.96 MB (hbm) | ≈ 4.6×tol | 1.28x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 297.44 ms / 312.75 ms | 6.174 s | 104.19 MB (hbm) | ≈ 21×tol | 2.94x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 560.74 ms / 978.33 ms | 1.481 s | 20.15 MB (hbm) | ✓ 2.7e-11×tol | 5.55x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax` | ok | 1.317 s / 1.364 s | 2.653 s | 1440 MB (rss) | ✓ 0.31×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 1.385 s / 1.423 s | 2.922 s | 1440 MB (rss) | ✓ 0.3×tol | 1.05x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-poly` | ok | 1.826 s / 1.866 s | 3.434 s | 1440 MB (rss) | ≈ 4.6×tol | 1.39x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 782.10 ms / 811.06 ms | 2.353 s | 1440 MB (rss) | ≈ 59×tol | 0.59x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 2.194 s / 2.485 s | 2.526 s | 1440 MB (rss) | ✓ 4.1e-11×tol | 1.67x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 139.38 ms / 365.10 ms | 677.00 ms | 80.35 MB (hbm) | ✓ 0.00027×tol | 0.63x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax` | ok | 222.19 ms / 257.05 ms | 4.146 s | 100.96 MB (hbm) | ✓ 0.34×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 222.46 ms / 300.32 ms | 5.275 s | 100.96 MB (hbm) | ✓ 0.33×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-poly` | ok | 364.06 ms / 416.18 ms | 4.579 s | 101.11 MB (hbm) | ≈ 4.6×tol | 1.64x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 138.45 ms / 142.65 ms | 5.138 s | 84.18 MB (hbm) | ≈ 59×tol | 0.62x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 6.947 s / 7.329 s | 7.487 s | 80.35 MB (hbm) | ✓ 4.4e-11×tol | 31.27x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

