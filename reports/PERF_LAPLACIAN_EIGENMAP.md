# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 461be9e91db11d8d05d7dd2d528beb5ba4a0dbb8 | bench: 267f82cceeba5036c25845a81eafa8a0411d098c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T06:24:31.079030+00:00

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
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `cupyx.sparse.eigsh` | ok | 93.43 ms / 108.86 ms | 514.48 ms | 4.19 MB (hbm) | ✓ 0.00035×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg` | ok | 88.62 ms / 89.48 ms | 3.538 s | 138.68 MB (hbm) | ✓ 0.016×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 88.16 ms / 88.22 ms | 4.334 s | 138.68 MB (hbm) | ✓ 0.018×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-poly` | ok | 48.90 ms / 48.97 ms | 4.094 s | 138.68 MB (hbm) | ✓ 0.53×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `nitrix-jax-shift_invert` | ok | 50.61 ms / 50.85 ms | 4.395 s | 105.13 MB (hbm) | ≈ 2.1×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=32,fmt=dense | `scipy.sparse.eigsh` | ok | 86.25 ms / 90.92 ms | 89.97 ms | 4.19 MB (hbm) | ✓ 3.2e-11×tol | — |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax` | ok | 91.90 ms / 98.20 ms | 438.37 ms | 1440 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 44.90 ms / 46.81 ms | 1.295 s | 1440 MB (rss) | ✓ 0.018×tol | 0.49x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 53.76 ms / 58.82 ms | 1.423 s | 1440 MB (rss) | ✓ 0.014×tol | 0.58x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 63.48 ms / 65.21 ms | 1.497 s | 1440 MB (rss) | ≈ 2.3×tol | 0.69x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 125.62 ms / 127.24 ms | 1.738 s | 1440 MB (rss) | ≈ 12×tol | 1.37x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 51.89 ms / 53.43 ms | 53.70 ms | 1440 MB (rss) | ✓ 1.7e-11×tol | 0.56x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 31.32 ms / 31.94 ms | 542.32 ms | 4.19 MB (hbm) | ✓ 0.00038×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 29.90 ms / 30.27 ms | 2.949 s | 138.49 MB (hbm) | ✓ 0.0091×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 29.40 ms / 29.92 ms | 3.720 s | 138.49 MB (hbm) | ✓ 0.012×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 16.87 ms / 17.87 ms | 4.625 s | 138.49 MB (hbm) | ≈ 2.4×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 23.12 ms / 23.39 ms | 3.574 s | 155.37 MB (hbm) | ≈ 11×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 52.71 ms / 58.42 ms | 52.97 ms | 4.19 MB (hbm) | ✓ 2.7e-11×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 849.27 ms / 1.203 s | 35.020 s | 8.39 MB (hbm) | n/a (no oracle) | 19.94x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 42.59 ms / 43.31 ms | 3.494 s | 143.34 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 42.37 ms / 42.61 ms | 4.162 s | 143.34 MB (hbm) | n/a (no oracle) | 0.99x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | ok | 609.07 ms / 664.52 ms | 729.30 ms | 8.39 MB (hbm) | n/a (no oracle) | 14.30x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 3.274 s / 4.145 s | 3.498 s | 67.11 MB (hbm) | n/a (no oracle) | 69.15x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 47.35 ms / 47.85 ms | 3.764 s | 204.47 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 48.08 ms / 48.63 ms | 4.307 s | 204.47 MB (hbm) | n/a (no oracle) | 1.02x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | timeout | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax` | ok | 527.67 ms / 542.18 ms | 666.90 ms | 1440 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 145.71 ms / 148.74 ms | 1.428 s | 1440 MB (rss) | ✓ 0.09×tol | 0.28x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 212.23 ms / 223.59 ms | 1.587 s | 1440 MB (rss) | ✓ 0.075×tol | 0.40x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 228.01 ms / 229.58 ms | 1.500 s | 1440 MB (rss) | ≈ 5.9×tol | 0.43x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 315.74 ms / 320.03 ms | 1.937 s | 1440 MB (rss) | ≈ 21×tol | 0.60x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 247.53 ms / 262.46 ms | 247.67 ms | 1440 MB (rss) | ✓ 3.6e-11×tol | 0.47x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 69.52 ms / 98.70 ms | 601.09 ms | 16.78 MB (hbm) | ✓ 0.00034×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 92.78 ms / 107.58 ms | 4.162 s | 117.44 MB (hbm) | ✓ 0.043×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 33.01 ms / 33.44 ms | 3.797 s | 121.64 MB (hbm) | ✓ 0.039×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 19.51 ms / 19.63 ms | 3.749 s | 117.96 MB (hbm) | ≈ 5.1×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 21.41 ms / 22.01 ms | 3.792 s | 151.14 MB (hbm) | ≈ 21×tol | — |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 270.25 ms / 314.04 ms | 245.16 ms | 16.78 MB (hbm) | ✓ 3.9e-11×tol | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 198.85 ms / 201.03 ms | 1.467 s | 1440 MB (rss) | ✓ 0.09×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 210.22 ms / 210.80 ms | 1.644 s | 1440 MB (rss) | ✓ 0.075×tol | 1.06x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 342.73 ms / 348.75 ms | 1.959 s | 1440 MB (rss) | ≈ 5.9×tol | 1.72x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 489.31 ms / 494.21 ms | 2.149 s | 1440 MB (rss) | ≈ 21×tol | 2.46x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 239.34 ms / 257.02 ms | 257.58 ms | 1440 MB (rss) | ✓ 2.7e-11×tol | 1.20x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 50.75 ms / 68.29 ms | 478.47 ms | 20.15 MB (hbm) | ✓ 0.0002×tol | 0.28x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 179.47 ms / 210.57 ms | 4.070 s | 120.96 MB (hbm) | ✓ 0.029×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 94.69 ms / 186.94 ms | 4.988 s | 120.96 MB (hbm) | ✓ 0.041×tol | 0.53x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 141.18 ms / 195.08 ms | 4.988 s | 120.96 MB (hbm) | ≈ 5.1×tol | 0.79x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 229.39 ms / 366.21 ms | 7.101 s | 104.19 MB (hbm) | ≈ 21×tol | 1.28x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 289.10 ms / 355.01 ms | 319.56 ms | 20.15 MB (hbm) | ✓ 2.4e-11×tol | 1.61x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax` | ok | 1.317 s / 1.364 s | 2.653 s | 1440 MB (rss) | ✓ 0.31×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 1.385 s / 1.423 s | 2.922 s | 1440 MB (rss) | ✓ 0.3×tol | 1.05x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-poly` | ok | 1.826 s / 1.866 s | 3.434 s | 1440 MB (rss) | ≈ 4.6×tol | 1.39x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 782.10 ms / 811.06 ms | 2.353 s | 1440 MB (rss) | ≈ 59×tol | 0.59x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=4096,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 2.194 s / 2.485 s | 2.526 s | 1440 MB (rss) | ✓ 4.1e-11×tol | 1.67x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 83.16 ms / 89.33 ms | 493.77 ms | 80.35 MB (hbm) | ✓ 0.00027×tol | 0.37x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax` | ok | 225.66 ms / 231.80 ms | 3.420 s | 100.96 MB (hbm) | ✓ 0.34×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 232.04 ms / 233.80 ms | 3.711 s | 100.96 MB (hbm) | ✓ 0.3×tol | 1.03x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-poly` | ok | 284.69 ms / 290.01 ms | 3.904 s | 101.11 MB (hbm) | ≈ 4.6×tol | 1.26x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 138.71 ms / 142.38 ms | 4.114 s | 84.18 MB (hbm) | ≈ 59×tol | 0.61x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=4096,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 2.594 s / 3.101 s | 3.109 s | 80.35 MB (hbm) | ✓ 4.4e-11×tol | 11.49x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 1.002 s / 1.128 s | 1.465 s | 33.55 MB (hbm) | n/a (no oracle) | 42.54x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 23.55 ms / 23.77 ms | 3.660 s | 153.94 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 45.12 ms / 46.98 ms | 4.677 s | 153.94 MB (hbm) | n/a (no oracle) | 1.92x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | ok | 5.621 s / 7.251 s | 26.120 s | 33.55 MB (hbm) | n/a (no oracle) | 238.67x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

