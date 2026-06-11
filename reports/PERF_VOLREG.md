# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c54bc81807fd0b81c371b5904a98e8e6f3d88a93 | bench: 7be151160d256117f2a68003be1befe98a76a202
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T18:50:31.315128+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.3dvolreg` | ok | 818.51 ms / 874.20 ms | 1.636 s | 1633 MB (rss) | n/a (no oracle) | 16.19x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.iofloor` | ok | 338.89 ms / 345.74 ms | 521.26 ms | 1633 MB (rss) | n/a (no oracle) | 6.70x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | ok | 2.794 s / 3.002 s | 6.569 s | 1633 MB (rss) | n/a (no oracle) | 55.26x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.iofloor` | ok | 373.33 ms / 384.44 ms | 490.75 ms | 1633 MB (rss) | n/a (no oracle) | 7.38x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.mcflirt` | ok | 530.00 ms / 535.91 ms | 651.34 ms | 1633 MB (rss) | n/a (no oracle) | 10.48x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 50.57 ms / 50.95 ms | 1.011 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 1.35 ms / 1.47 ms | 3.538 s | 136.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.3dvolreg` | ok | 1.472 s / 1.497 s | 1.718 s | 1633 MB (rss) | n/a (no oracle) | 11.77x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.iofloor` | ok | 494.33 ms / 497.09 ms | 639.34 ms | 1633 MB (rss) | n/a (no oracle) | 3.95x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | ok | 6.694 s / 7.747 s | 11.349 s | 1633 MB (rss) | n/a (no oracle) | 53.53x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.iofloor` | ok | 619.10 ms / 671.66 ms | 801.16 ms | 1633 MB (rss) | n/a (no oracle) | 4.95x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.mcflirt` | ok | 828.05 ms / 834.97 ms | 1.038 s | 1633 MB (rss) | n/a (no oracle) | 6.62x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 125.03 ms / 128.13 ms | 1.012 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 3.02 ms / 3.08 ms | 3.564 s | 138.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.3dvolreg` | ok | 653.70 ms / 654.88 ms | 871.83 ms | 1633 MB (rss) | n/a (no oracle) | 23.13x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.iofloor` | ok | 223.86 ms / 231.58 ms | 388.81 ms | 1633 MB (rss) | n/a (no oracle) | 7.92x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | ok | 1.241 s / 1.262 s | 6.137 s | 1633 MB (rss) | n/a (no oracle) | 43.92x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.iofloor` | ok | 254.04 ms / 274.15 ms | 386.92 ms | 1633 MB (rss) | n/a (no oracle) | 8.99x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.mcflirt` | ok | 368.71 ms / 388.56 ms | 850.99 ms | 1633 MB (rss) | n/a (no oracle) | 13.05x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 28.26 ms / 31.13 ms | 773.75 ms | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 1.02 ms / 1.03 ms | 3.625 s | 135.53 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `afni.3dvolreg` | ok | 9.345 s / 9.571 s | 9.515 s | 1633 MB (rss) | n/a (no oracle) | 2.46x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `afni.iofloor` | ok | 3.890 s / 3.937 s | 4.034 s | 1633 MB (rss) | n/a (no oracle) | 1.02x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 22.421 s / 23.034 s | 26.216 s | 1633 MB (rss) | n/a (no oracle) | 5.90x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `fsl.iofloor` | ok | 4.199 s / 4.202 s | 4.454 s | 1633 MB (rss) | n/a (no oracle) | 1.11x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `fsl.mcflirt` | ok | 6.209 s / 6.273 s | 6.386 s | 1633 MB (rss) | n/a (no oracle) | 1.63x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 3.799 s / 3.907 s | 6.260 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 172.76 ms / 172.97 ms | 10.002 s | 625.34 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `afni.3dvolreg` | ok | 19.045 s / 19.132 s | 19.713 s | 1633 MB (rss) | n/a (no oracle) | 2.42x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `afni.iofloor` | ok | 7.682 s / 7.732 s | 7.941 s | 1633 MB (rss) | n/a (no oracle) | 0.97x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `ants.motion_correction` | ok | 58.368 s / 62.092 s | 62.269 s | 1633 MB (rss) | n/a (no oracle) | 7.41x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `fsl.iofloor` | ok | 8.274 s / 8.338 s | 8.431 s | 1633 MB (rss) | n/a (no oracle) | 1.05x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `fsl.mcflirt` | ok | 12.191 s / 12.211 s | 12.559 s | 1633 MB (rss) | n/a (no oracle) | 1.55x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `nitrix-jax` | ok | 7.882 s / 8.195 s | 11.024 s | 1719 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `nitrix-jax` | ok | 355.58 ms / 355.64 ms | 10.135 s | 1207.97 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `afni.3dvolreg` | ok | 4.773 s / 4.820 s | 5.910 s | 1633 MB (rss) | n/a (no oracle) | 2.50x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `afni.iofloor` | ok | 2.079 s / 2.093 s | 2.196 s | 1633 MB (rss) | n/a (no oracle) | 1.09x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `ants.motion_correction` | ok | 11.073 s / 11.454 s | 18.177 s | 1633 MB (rss) | n/a (no oracle) | 5.80x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `fsl.iofloor` | ok | 2.219 s / 2.253 s | 2.447 s | 1633 MB (rss) | n/a (no oracle) | 1.16x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `fsl.mcflirt` | ok | 3.234 s / 3.305 s | 3.526 s | 1633 MB (rss) | n/a (no oracle) | 1.69x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `nitrix-jax` | ok | 1.909 s / 1.926 s | 3.769 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `nitrix-jax` | ok | 81.24 ms / 81.28 ms | 6.695 s | 334.79 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `afni.3dvolreg` | ok | 46.875 s / 46.976 s | 47.695 s | 1789 MB (rss) | n/a (no oracle) | 2.16x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `afni.iofloor` | ok | 18.865 s / 18.944 s | 18.543 s | 1820 MB (rss) | n/a (no oracle) | 0.87x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `ants.motion_correction` | timeout | — | — | — | — | — |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `fsl.iofloor` | ok | 19.601 s / 19.614 s | 20.164 s | 1781 MB (rss) | n/a (no oracle) | 0.91x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `fsl.mcflirt` | ok | 29.766 s / 30.068 s | 30.096 s | 1789 MB (rss) | n/a (no oracle) | 1.37x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `nitrix-jax` | ok | 21.658 s / 21.688 s | 24.831 s | 3070 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `nitrix-jax` | ok | 904.76 ms / 904.86 ms | 12.492 s | 2815.25 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `afni.3dvolreg` | ok | 19.262 s / 19.455 s | 19.515 s | 1633 MB (rss) | n/a (no oracle) | 1.67x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `afni.iofloor` | ok | 8.840 s / 8.847 s | 9.049 s | 1633 MB (rss) | n/a (no oracle) | 0.77x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 30.864 s / 31.191 s | 40.311 s | 1633 MB (rss) | n/a (no oracle) | 2.68x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `fsl.iofloor` | ok | 9.601 s / 9.655 s | 9.947 s | 1633 MB (rss) | n/a (no oracle) | 0.83x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `fsl.mcflirt` | ok | 14.232 s / 14.284 s | 14.400 s | 1633 MB (rss) | n/a (no oracle) | 1.24x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 11.509 s / 11.567 s | 13.949 s | 1810 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 443.45 ms / 443.47 ms | 10.039 s | 1347.45 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `afni.3dvolreg` | ok | 32.689 s / 32.726 s | 32.673 s | 1650 MB (rss) | n/a (no oracle) | 1.26x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `afni.iofloor` | ok | 16.860 s / 16.926 s | 16.972 s | 1650 MB (rss) | n/a (no oracle) | 0.65x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 36.268 s / 42.598 s | 49.087 s | 1997 MB (rss) | n/a (no oracle) | 1.40x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `fsl.iofloor` | ok | 18.117 s / 18.151 s | 18.246 s | 1840 MB (rss) | n/a (no oracle) | 0.70x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `fsl.mcflirt` | ok | 26.538 s / 26.555 s | 26.777 s | 1650 MB (rss) | n/a (no oracle) | 1.02x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 25.982 s / 26.314 s | 28.722 s | 2884 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 1.073 s / 1.074 s | 12.092 s | 2638.00 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

