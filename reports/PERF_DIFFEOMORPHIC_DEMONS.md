# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ab52d0b209fcd3ff667d4013b759c79341728407 | bench: 935ca7ff685b314c4bdfb21967bd3349bd6ad0e9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-17T03:14:58.136945+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | ok | 4.783 s / 4.863 s | 10.268 s | 863 MB (rss) | n/a (no oracle) | 0.46x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | ok | 39.819 s / 40.375 s | 45.680 s | 863 MB (rss) | n/a (no oracle) | 3.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 10.294 s / 11.249 s | 16.157 s | 1143 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax-algebra` | ok | 14.967 s / 15.104 s | 19.211 s | 1104 MB (rss) | n/a (no oracle) | 1.45x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `simpleitk.demons` | ok | 2.797 s / 2.941 s | 2.803 s | 863 MB (rss) | n/a (no oracle) | 0.27x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 69.31 ms / 73.13 ms | 31.682 s | 1634.75 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax-algebra` | ok | 88.16 ms / 88.36 ms | 27.188 s | 1634.75 MB (hbm) | n/a (no oracle) | 1.27x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `simpleitk.demons` | ok | 3.127 s / 4.009 s | 4.188 s | 16.78 MB (hbm) | n/a (no oracle) | 45.11x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 8.870 s / 9.014 s | 12.858 s | 863 MB (rss) | n/a (no oracle) | 0.36x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | ok | 126.992 s / 127.789 s | 127.181 s | 923 MB (rss) | n/a (no oracle) | 5.14x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 24.715 s / 29.709 s | 31.515 s | 1272 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 17.634 s / 17.982 s | 22.291 s | 1116 MB (rss) | n/a (no oracle) | 0.71x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 6.105 s / 6.133 s | 6.085 s | 923 MB (rss) | n/a (no oracle) | 0.25x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 281.27 ms / 281.60 ms | 27.967 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 221.44 ms / 221.68 ms | 18.939 s | 8758.24 MB (hbm) | n/a (no oracle) | 0.79x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 10.567 s / 11.175 s | 11.762 s | 16.78 MB (hbm) | n/a (no oracle) | 37.57x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 15.653 s / 15.812 s | 20.167 s | 863 MB (rss) | n/a (no oracle) | 0.67x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | ok | 83.891 s / 85.743 s | 89.761 s | 970 MB (rss) | n/a (no oracle) | 3.61x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 23.240 s / 26.848 s | 30.940 s | 1368 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax-algebra` | ok | 23.872 s / 26.373 s | 30.045 s | 1210 MB (rss) | n/a (no oracle) | 1.03x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 6.167 s / 6.181 s | 5.906 s | 933 MB (rss) | n/a (no oracle) | 0.27x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 183.47 ms / 184.11 ms | 28.332 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax-algebra` | ok | 228.82 ms / 228.98 ms | 19.334 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.25x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 10.549 s / 10.985 s | 11.909 s | 16.78 MB (hbm) | n/a (no oracle) | 57.49x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 17.426 s / 20.837 s | 22.653 s | 1231 MB (rss) | n/a (no oracle) | 0.34x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | timeout | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 51.539 s / 56.932 s | 69.169 s | 1632 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 30.623 s / 31.168 s | 35.265 s | 1490 MB (rss) | n/a (no oracle) | 0.59x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 11.903 s / 12.302 s | 13.002 s | 1273 MB (rss) | n/a (no oracle) | 0.23x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 741.09 ms / 741.18 ms | 24.238 s | 13643.81 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 497.67 ms / 497.68 ms | 14.571 s | 13643.81 MB (hbm) | n/a (no oracle) | 0.67x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 19.499 s / 22.455 s | 18.563 s | 33.55 MB (hbm) | n/a (no oracle) | 26.31x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | ok | 1.510 s / 1.512 s | 8.069 s | 863 MB (rss) | n/a (no oracle) | 1.63x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | ok | 3.231 s / 3.239 s | 4.295 s | 863 MB (rss) | n/a (no oracle) | 3.48x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 929.14 ms / 934.70 ms | 3.262 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax-algebra` | ok | 2.737 s / 3.606 s | 4.332 s | 863 MB (rss) | n/a (no oracle) | 2.95x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 647.09 ms / 934.22 ms | 1.358 s | 863 MB (rss) | n/a (no oracle) | 0.70x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 7.61 ms / 8.38 ms | 13.320 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax-algebra` | ok | 4.22 ms / 4.22 ms | 6.763 s | 203.42 MB (hbm) | n/a (no oracle) | 0.55x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 2.169 s / 2.456 s | 1.304 s | 0.88 MB (hbm) | n/a (no oracle) | 285.13x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 1.683 s / 1.751 s | 7.573 s | 863 MB (rss) | n/a (no oracle) | 1.29x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 5.403 s / 5.439 s | 5.772 s | 863 MB (rss) | n/a (no oracle) | 4.14x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 1.305 s / 1.464 s | 12.631 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 2.541 s / 3.109 s | 9.641 s | 863 MB (rss) | n/a (no oracle) | 1.95x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 597.58 ms / 772.19 ms | 683.74 ms | 863 MB (rss) | n/a (no oracle) | 0.46x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 8.17 ms / 8.79 ms | 19.089 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 6.65 ms / 7.02 ms | 9.015 s | 203.42 MB (hbm) | n/a (no oracle) | 0.81x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 1.171 s / 1.897 s | 1.156 s | 0.88 MB (hbm) | n/a (no oracle) | 143.36x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | ok | 1.687 s / 1.895 s | 6.603 s | 863 MB (rss) | n/a (no oracle) | 0.90x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | ok | 8.355 s / 8.402 s | 10.155 s | 863 MB (rss) | n/a (no oracle) | 4.46x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 1.874 s / 1.894 s | 10.794 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax-algebra` | ok | 4.087 s / 4.576 s | 8.850 s | 863 MB (rss) | n/a (no oracle) | 2.18x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 641.68 ms / 657.02 ms | 689.38 ms | 863 MB (rss) | n/a (no oracle) | 0.34x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 12.63 ms / 12.70 ms | 15.968 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax-algebra` | ok | 12.29 ms / 12.47 ms | 10.057 s | 203.42 MB (hbm) | n/a (no oracle) | 0.97x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 1.270 s / 1.789 s | 1.042 s | 0.88 MB (hbm) | n/a (no oracle) | 100.55x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 4.481 s / 4.828 s | 8.714 s | 863 MB (rss) | n/a (no oracle) | 0.40x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | ok | 35.538 s / 35.545 s | 36.481 s | 863 MB (rss) | n/a (no oracle) | 3.15x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 11.287 s / 11.436 s | 18.554 s | 971 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 23.577 s / 24.531 s | 37.844 s | 946 MB (rss) | n/a (no oracle) | 2.09x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 3.578 s / 3.677 s | 4.049 s | 863 MB (rss) | n/a (no oracle) | 0.32x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 66.02 ms / 69.50 ms | 24.454 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 59.69 ms / 60.96 ms | 14.876 s | 1309.54 MB (hbm) | n/a (no oracle) | 0.90x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 4.381 s / 4.475 s | 4.700 s | 8.39 MB (hbm) | n/a (no oracle) | 66.35x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 6.846 s / 6.861 s | 12.320 s | 863 MB (rss) | n/a (no oracle) | 0.30x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | ok | 35.712 s / 36.221 s | 37.622 s | 863 MB (rss) | n/a (no oracle) | 1.55x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 22.986 s / 24.877 s | 26.429 s | 1026 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax-algebra` | ok | 33.952 s / 34.340 s | 39.384 s | 1015 MB (rss) | n/a (no oracle) | 1.48x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 2.428 s / 2.590 s | 2.534 s | 863 MB (rss) | n/a (no oracle) | 0.11x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 51.29 ms / 52.87 ms | 23.784 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax-algebra` | ok | 65.06 ms / 66.70 ms | 21.520 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.27x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 8.041 s / 8.230 s | 8.774 s | 8.39 MB (hbm) | n/a (no oracle) | 156.80x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

