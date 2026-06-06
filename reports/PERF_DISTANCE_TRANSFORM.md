# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: 44def4b7ce5c1f37844a65f1545ddc4ba9281c5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T22:38:09.061263+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform | jax-cpu | shape=[128, 128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[128, 128, 128] | `nitrix-jax` | ok | 165.48 ms / 182.50 ms | 412.77 ms | 1681 MB (rss) | ✓ 5.7e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 337.78 ms / 370.94 ms | 350.41 ms | 1681 MB (rss) | ✓ 0×tol | 2.04x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 686.72 ms / 687.82 ms | 756.56 ms | 1681 MB (rss) | ≈ 2.8e+02×tol | 4.15x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 534.2 µs / 549.4 µs | 233.93 ms | 8.39 MB (hbm) | ✓ 0×tol | 1.08x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 496.2 µs / 506.8 µs | 1.174 s | 58.72 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 313.12 ms / 331.97 ms | 335.69 ms | 8.39 MB (hbm) | ✓ 0×tol | 631.03x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 691.20 ms / 716.32 ms | 765.72 ms | 8.39 MB (hbm) | ≈ 2.8e+02×tol | 1392.95x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 25.10 ms / 29.01 ms | 228.75 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 917.6 µs / 979.6 µs | 1.43 ms | 712 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.92 ms / 1.95 ms | 61.37 ms | 712 MB (rss) | ✓ 3.2e-08×tol | 0.08x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 201.7 µs / 208.0 µs | 127.89 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 4.99 ms / 5.00 ms | 562.98 ms | 68.35 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 981.7 µs / 1.37 ms | 1.52 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.20x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.95 ms / 2.11 ms | 59.24 ms | 0.07 MB (hbm) | ✓ 3.2e-08×tol | 0.39x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[256, 256, 256] | `nitrix-jax` | ok | 3.207 s / 3.706 s | 5.433 s | 1681 MB (rss) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 3.833 s / 4.300 s | 4.214 s | 1723 MB (rss) | ✓ 0×tol | 1.20x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 5.789 s / 5.936 s | 5.727 s | 1681 MB (rss) | ≈ 3.9e+02×tol | 1.81x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 6.00 ms / 6.09 ms | 320.75 ms | 67.11 MB (hbm) | ✓ 0×tol | 0.82x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `nitrix-jax` | ok | 7.35 ms / 8.02 ms | 1.514 s | 335.54 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 3.727 s / 4.071 s | 4.986 s | 67.11 MB (hbm) | ✓ 0×tol | 507.36x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 5.857 s / 5.928 s | 5.964 s | 67.11 MB (hbm) | ≈ 3.9e+02×tol | 797.29x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[256, 256] | `nitrix-jax` | ok | 7.50 ms / 8.33 ms | 162.73 ms | 1681 MB (rss) | ✓ 5.7e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 2.77 ms / 3.07 ms | 2.97 ms | 1681 MB (rss) | ✓ 0×tol | 0.37x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 5.98 ms / 6.01 ms | 71.93 ms | 1681 MB (rss) | ≈ 59×tol | 0.80x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 246.2 µs / 250.3 µs | 522.20 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.72x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `nitrix-jax` | ok | 343.7 µs / 352.7 µs | 1.166 s | 34.34 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `scipy.ndimage.distance_transform_edt` | ok | 2.59 ms / 3.01 ms | 4.80 ms | 0.26 MB (hbm) | ✓ 0×tol | 7.53x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[256, 256] | `simpleitk.DanielssonDistanceMap` | ok | 6.03 ms / 6.11 ms | 64.27 ms | 0.26 MB (hbm) | ≈ 59×tol | 17.55x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `nitrix-jax` | ok | 85.78 ms / 87.54 ms | 314.87 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.51 ms / 4.41 ms | 4.91 ms | 712 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 10.14 ms / 10.48 ms | 67.56 ms | 712 MB (rss) | ✓ 2.4e-08×tol | 0.12x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 146.9 µs / 168.5 µs | 155.57 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `nitrix-jax` | ok | 3.91 ms / 3.92 ms | 609.23 ms | 74.32 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.37 ms / 4.52 ms | 5.12 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.86x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 10.21 ms / 10.36 ms | 70.34 ms | 0.13 MB (hbm) | ✓ 2.4e-08×tol | 2.61x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32] | `nitrix-jax` | ok | 426.8 µs / 446.8 µs | 173.86 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 58.3 µs / 70.2 µs | 103.2 µs | 712 MB (rss) | ✓ 0×tol | 0.14x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 565.1 µs / 592.0 µs | 56.73 ms | 712 MB (rss) | ✓ 2.4e-08×tol | 1.32x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 184.3 µs / 191.3 µs | 164.52 ms | 0.00 MB (hbm) | ✓ 0×tol | 0.11x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `nitrix-jax` | ok | 1.62 ms / 1.66 ms | 547.11 ms | 33.63 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 81.0 µs / 90.7 µs | 129.5 µs | 0.00 MB (hbm) | ✓ 0×tol | 0.05x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 553.3 µs / 566.8 µs | 57.50 ms | 0.00 MB (hbm) | ✓ 2.4e-08×tol | 0.34x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[512, 512] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[512, 512] | `nitrix-jax` | ok | 62.76 ms / 67.47 ms | 228.44 ms | 1681 MB (rss) | ✓ 5.7e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[512, 512] | `scipy.ndimage.distance_transform_edt` | ok | 11.29 ms / 16.06 ms | 13.33 ms | 1681 MB (rss) | ✓ 0×tol | 0.18x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[512, 512] | `simpleitk.DanielssonDistanceMap` | ok | 22.52 ms / 22.69 ms | 78.95 ms | 1681 MB (rss) | ≈ 59×tol | 0.36x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 330.9 µs / 334.9 µs | 182.62 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.57x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `nitrix-jax` | ok | 575.5 µs / 581.9 µs | 824.56 ms | 36.70 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `scipy.ndimage.distance_transform_edt` | ok | 11.75 ms / 17.96 ms | 13.47 ms | 1.05 MB (hbm) | ✓ 0×tol | 20.42x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[512, 512] | `simpleitk.DanielssonDistanceMap` | ok | 22.53 ms / 22.65 ms | 79.26 ms | 1.05 MB (hbm) | ≈ 59×tol | 39.15x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 16.13 ms / 17.24 ms | 244.23 ms | 1681 MB (rss) | ✓ 5.4e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 28.98 ms / 33.24 ms | 39.37 ms | 1681 MB (rss) | ✓ 0×tol | 1.80x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 81.26 ms / 82.10 ms | 138.94 ms | 1681 MB (rss) | ≈ 3.9e+02×tol | 5.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 232.8 µs / 235.5 µs | 194.24 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.97x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 239.5 µs / 247.4 µs | 907.33 ms | 36.70 MB (hbm) | ✓ 9.1e-05×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 29.95 ms / 40.52 ms | 34.60 ms | 1.05 MB (hbm) | ✓ 0×tol | 125.06x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 81.28 ms / 81.97 ms | 143.72 ms | 1.05 MB (hbm) | ≈ 3.9e+02×tol | 339.37x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

