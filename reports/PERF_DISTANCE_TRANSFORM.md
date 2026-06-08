# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: bf29b0e04a10f2539ace538371bad011340aa9fd | bench: b96475991d7df9a474051537d6bb14ba2a6b078c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T00:30:57.285280+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 540.0 µs / 559.3 µs | 176.91 ms | 8.39 MB (hbm) | ✓ 0×tol | 1.15x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 469.6 µs / 476.9 µs | 875.74 ms | 58.72 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 310.68 ms / 324.38 ms | 339.63 ms | 8.39 MB (hbm) | ✓ 0×tol | 661.59x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 705.26 ms / 719.08 ms | 785.04 ms | 8.39 MB (hbm) | ≈ 2.8e+02×tol | 1501.81x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=16 | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 10.78 ms / 10.79 ms | 226.54 ms | 134.22 MB (hbm) | ✓ 0×tol | 1.10x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=16 | `nitrix-jax` | ok | 9.76 ms / 10.28 ms | 925.79 ms | 671.09 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=16 | `scipy.ndimage.distance_transform_edt` | ok | 5.229 s / 5.311 s | 5.264 s | 134.22 MB (hbm) | ✓ 0×tol | 535.63x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=4 | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 2.68 ms / 2.68 ms | 210.77 ms | 33.55 MB (hbm) | ✓ 0×tol | 1.20x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=4 | `nitrix-jax` | ok | 2.24 ms / 2.24 ms | 897.34 ms | 167.77 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=4 | `scipy.ndimage.distance_transform_edt` | ok | 1.340 s / 1.350 s | 1.368 s | 33.55 MB (hbm) | ✓ 0×tol | 599.45x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=8 | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 5.39 ms / 5.41 ms | 273.75 ms | 67.11 MB (hbm) | ✓ 0×tol | 1.09x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=8 | `nitrix-jax` | ok | 4.96 ms / 5.02 ms | 873.43 ms | 335.54 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128],batch=8 | `scipy.ndimage.distance_transform_edt` | ok | 2.589 s / 2.653 s | 2.629 s | 67.11 MB (hbm) | ✓ 0×tol | 522.34x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 1.63 ms / 1.65 ms | 139.92 ms | 713 MB (rss) | ✓ 5.2e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 621.9 µs / 636.5 µs | 725.0 µs | 713 MB (rss) | ✓ 0×tol | 0.38x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.85 ms / 1.89 ms | 59.62 ms | 713 MB (rss) | ≈ 59×tol | 1.13x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 200.9 µs / 212.2 µs | 192.14 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.92x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 219.1 µs / 221.4 µs | 804.86 ms | 33.75 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 626.2 µs / 779.8 µs | 848.4 µs | 0.07 MB (hbm) | ✓ 0×tol | 2.86x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.88 ms / 1.92 ms | 59.39 ms | 0.07 MB (hbm) | ≈ 59×tol | 8.56x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 6.05 ms / 6.09 ms | 234.60 ms | 67.11 MB (hbm) | ✓ 0×tol | 0.83x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `nitrix-jax` | ok | 7.31 ms / 7.86 ms | 1.007 s | 335.54 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 3.172 s / 3.197 s | 3.203 s | 67.11 MB (hbm) | ✓ 0×tol | 433.90x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 5.657 s / 5.900 s | 6.017 s | 67.11 MB (hbm) | ≈ 3.9e+02×tol | 773.71x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 244.3 µs / 249.3 µs | 166.82 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.72x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `nitrix-jax` | ok | 341.2 µs / 349.4 µs | 776.41 ms | 34.34 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 2.55 ms / 2.59 ms | 3.49 ms | 0.26 MB (hbm) | ✓ 0×tol | 7.47x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 6.08 ms / 6.17 ms | 66.67 ms | 0.26 MB (hbm) | ≈ 59×tol | 17.81x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 332.7 µs / 340.8 µs | 199.72 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.59x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `nitrix-jax` | ok | 564.2 µs / 572.0 µs | 729.82 ms | 36.70 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `scipy.ndimage.distance_transform_edt` | ok | 11.08 ms / 12.61 ms | 13.45 ms | 1.05 MB (hbm) | ✓ 0×tol | 19.64x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `simpleitk.DanielssonDistanceMap` | ok | 22.80 ms / 23.03 ms | 77.97 ms | 1.05 MB (hbm) | ≈ 59×tol | 40.40x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 12.81 ms / 13.01 ms | 183.96 ms | 713 MB (rss) | ✓ 5.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 28.45 ms / 28.82 ms | 33.77 ms | 713 MB (rss) | ✓ 0×tol | 2.22x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 81.16 ms / 81.29 ms | 138.90 ms | 713 MB (rss) | ≈ 3.9e+02×tol | 6.34x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 235.9 µs / 239.6 µs | 173.81 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.96x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 245.1 µs / 254.0 µs | 826.74 ms | 36.70 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 29.29 ms / 31.13 ms | 35.79 ms | 1.05 MB (hbm) | ✓ 0×tol | 119.51x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 81.60 ms / 82.21 ms | 138.02 ms | 1.05 MB (hbm) | ≈ 3.9e+02×tol | 332.90x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[64, 64] | `nitrix-jax` | ok | 261.1 µs / 268.3 µs | 151.75 ms | 713 MB (rss) | ✓ 4.9e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 170.1 µs / 175.6 µs | 229.1 µs | 713 MB (rss) | ✓ 0×tol | 0.65x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 854.4 µs / 875.6 µs | 58.08 ms | 713 MB (rss) | ≈ 59×tol | 3.27x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 199.9 µs / 202.9 µs | 356.84 ms | 0.02 MB (hbm) | ✓ 0×tol | 1.34x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64] | `nitrix-jax` | ok | 149.8 µs / 165.9 µs | 1.128 s | 33.60 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 297.7 µs / 312.3 µs | 296.0 µs | 0.02 MB (hbm) | ✓ 0×tol | 1.99x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 845.4 µs / 861.4 µs | 65.97 ms | 0.02 MB (hbm) | ≈ 59×tol | 5.64x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

