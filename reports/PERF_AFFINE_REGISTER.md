# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c54bc81807fd0b81c371b5904a98e8e6f3d88a93 | bench: 7be151160d256117f2a68003be1befe98a76a202
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T17:21:33.063895+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| affine_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | ok | 270.03 ms / 274.31 ms | 3.310 s | 857 MB (rss) | n/a (no oracle) | 0.38x vs nitrix-jax |
| affine_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | ok | 12.780 s / 12.819 s | 14.771 s | 857 MB (rss) | n/a (no oracle) | 18.00x vs nitrix-jax |
| affine_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 710.02 ms / 717.50 ms | 5.707 s | 896 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 36.59 ms / 36.59 ms | 10.625 s | 1634.75 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 1.149 s / 1.244 s | 4.725 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | ok | 25.116 s / 25.135 s | 26.154 s | 857 MB (rss) | n/a (no oracle) | 21.89x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 1.147 s / 1.155 s | 4.794 s | 1048 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 69.68 ms / 69.92 ms | 12.186 s | 8733.60 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 470.28 ms / 1.303 s | 5.148 s | 857 MB (rss) | n/a (no oracle) | 0.06x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | ok | 26.247 s / 26.308 s | 27.292 s | 857 MB (rss) | n/a (no oracle) | 3.26x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 8.040 s / 8.068 s | 17.802 s | 1417 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 198.12 ms / 198.57 ms | 20.509 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 1.461 s / 1.640 s | 5.409 s | 857 MB (rss) | n/a (no oracle) | 0.64x vs nitrix-jax |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | ok | 82.632 s / 82.730 s | 83.661 s | 857 MB (rss) | n/a (no oracle) | 36.09x vs nitrix-jax |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 2.290 s / 2.331 s | 5.881 s | 1381 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 138.91 ms / 139.01 ms | 12.760 s | 13694.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | ok | 980.24 ms / 999.17 ms | 5.060 s | 857 MB (rss) | n/a (no oracle) | 0.25x vs nitrix-jax |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | ok | 88.775 s / 88.822 s | 89.703 s | 1055 MB (rss) | n/a (no oracle) | 22.61x vs nitrix-jax |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 3.926 s / 3.955 s | 7.603 s | 1853 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 244.58 ms / 245.25 ms | 28.864 s | 1374.56 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 151.82 ms / 163.87 ms | 5.161 s | 857 MB (rss) | n/a (no oracle) | 6.88x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | ok | 552.06 ms / 553.31 ms | 1.371 s | 857 MB (rss) | n/a (no oracle) | 25.03x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 22.06 ms / 26.65 ms | 2.600 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 4.70 ms / 5.42 ms | 5.255 s | 218.99 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 149.66 ms / 152.44 ms | 3.314 s | 857 MB (rss) | n/a (no oracle) | 2.11x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 1.227 s / 1.228 s | 1.819 s | 857 MB (rss) | n/a (no oracle) | 17.27x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 71.03 ms / 73.98 ms | 4.273 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 7.65 ms / 7.89 ms | 8.342 s | 218.99 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 696.51 ms / 915.20 ms | 4.082 s | 857 MB (rss) | n/a (no oracle) | 9.10x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | ok | 1.409 s / 1.410 s | 1.973 s | 857 MB (rss) | n/a (no oracle) | 18.41x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 76.51 ms / 77.66 ms | 5.008 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 11.49 ms / 11.76 ms | 11.068 s | 218.99 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 520.83 ms / 904.28 ms | 5.698 s | 857 MB (rss) | n/a (no oracle) | 1.13x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | ok | 11.849 s / 11.856 s | 12.615 s | 857 MB (rss) | n/a (no oracle) | 25.72x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 460.71 ms / 474.06 ms | 4.422 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 26.36 ms / 26.41 ms | 10.137 s | 1279.82 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 273.72 ms / 622.55 ms | 5.665 s | 857 MB (rss) | n/a (no oracle) | 0.07x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | ok | 19.374 s / 19.434 s | 20.259 s | 857 MB (rss) | n/a (no oracle) | 5.27x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 3.673 s / 4.074 s | 11.714 s | 1077 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 78.45 ms / 78.52 ms | 14.998 s | 1484.80 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

