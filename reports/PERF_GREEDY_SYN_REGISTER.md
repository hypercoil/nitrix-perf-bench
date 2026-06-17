# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 60e919f9975e3dd0eb7e15b0fe34988d9c260447 | bench: 935ca7ff685b314c4bdfb21967bd3349bd6ad0e9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-16T22:57:59.408618+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `ants.registration` | ok | 6.342 s / 6.725 s | 12.467 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `dipy.registration` | ok | 153.338 s / 153.611 s | 155.739 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax-mi` | ok | 91.361 s / 92.706 s | 118.447 s | 1465 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 94.706 s / 95.627 s | 101.961 s | 1494 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax` | ok | 534.62 ms / 536.01 ms | 44.869 s | 1634.75 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax-mi` | ok | 1.244 s / 1.249 s | 46.370 s | 1634.75 MB (hbm) | n/a (no oracle) | 2.33x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 1.247 s / 1.249 s | 52.697 s | 1634.75 MB (hbm) | n/a (no oracle) | 2.33x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | ok | 6.709 s / 7.260 s | 13.313 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | ok | 112.009 s / 112.379 s | 124.260 s | 961 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax-mi` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 176.783 s / 204.294 s | 226.661 s | 1847 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | ok | 1.236 s / 1.237 s | 40.635 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax-mi` | ok | 1.468 s / 1.469 s | 38.682 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.19x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 1.477 s / 1.480 s | 35.244 s | 8758.24 MB (hbm) | n/a (no oracle) | 1.19x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | ok | 1.306 s / 1.318 s | 6.561 s | 815 MB (rss) | n/a (no oracle) | 0.09x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | ok | 4.796 s / 4.946 s | 7.758 s | 815 MB (rss) | n/a (no oracle) | 0.32x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 14.803 s / 15.297 s | 21.021 s | 815 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi` | ok | 6.756 s / 6.856 s | 11.826 s | 815 MB (rss) | n/a (no oracle) | 0.46x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi-autodiff` | ok | 3.435 s / 3.503 s | 6.860 s | 815 MB (rss) | n/a (no oracle) | 0.23x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 21.49 ms / 21.63 ms | 13.826 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi` | ok | 28.89 ms / 28.98 ms | 10.997 s | 291.85 MB (hbm) | n/a (no oracle) | 1.34x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi-autodiff` | ok | 29.15 ms / 29.33 ms | 11.901 s | 291.85 MB (hbm) | n/a (no oracle) | 1.36x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | ok | 1.481 s / 1.483 s | 7.228 s | 815 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | ok | 11.780 s / 11.791 s | 13.233 s | 815 MB (rss) | n/a (no oracle) | 0.30x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 38.654 s / 39.469 s | 57.536 s | 969 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi` | ok | 13.951 s / 14.684 s | 25.289 s | 821 MB (rss) | n/a (no oracle) | 0.36x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 9.188 s / 9.317 s | 19.259 s | 815 MB (rss) | n/a (no oracle) | 0.24x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 64.35 ms / 64.40 ms | 35.613 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi` | ok | 80.69 ms / 80.81 ms | 28.572 s | 291.85 MB (hbm) | n/a (no oracle) | 1.25x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 82.93 ms / 83.25 ms | 24.089 s | 291.85 MB (hbm) | n/a (no oracle) | 1.29x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | ok | 1.292 s / 1.333 s | 5.126 s | 815 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | ok | 10.583 s / 10.736 s | 11.699 s | 815 MB (rss) | n/a (no oracle) | 0.33x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 31.798 s / 32.189 s | 51.874 s | 1146 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi` | ok | 11.208 s / 12.282 s | 26.378 s | 994 MB (rss) | n/a (no oracle) | 0.35x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi-autodiff` | ok | 7.968 s / 8.102 s | 18.822 s | 1010 MB (rss) | n/a (no oracle) | 0.25x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 74.60 ms / 74.68 ms | 39.076 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi` | ok | 91.98 ms / 92.09 ms | 30.632 s | 291.85 MB (hbm) | n/a (no oracle) | 1.23x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi-autodiff` | ok | 95.44 ms / 102.36 ms | 37.417 s | 291.85 MB (hbm) | n/a (no oracle) | 1.28x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | ok | 1.733 s / 1.748 s | 6.917 s | 815 MB (rss) | n/a (no oracle) | 0.02x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | ok | 13.075 s / 13.095 s | 14.397 s | 815 MB (rss) | n/a (no oracle) | 0.11x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 114.541 s / 122.343 s | 137.107 s | 1073 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax-mi` | ok | 29.581 s / 31.927 s | 43.060 s | 934 MB (rss) | n/a (no oracle) | 0.26x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 22.988 s / 23.179 s | 32.653 s | 924 MB (rss) | n/a (no oracle) | 0.20x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 115.54 ms / 115.60 ms | 29.668 s | 616.70 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax-mi` | ok | 149.71 ms / 149.86 ms | 28.496 s | 616.70 MB (hbm) | n/a (no oracle) | 1.30x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 151.82 ms / 151.87 ms | 26.938 s | 616.70 MB (hbm) | n/a (no oracle) | 1.31x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 2.798 s / 2.817 s | 8.655 s | 815 MB (rss) | n/a (no oracle) | 0.02x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | ok | 32.248 s / 32.319 s | 34.954 s | 815 MB (rss) | n/a (no oracle) | 0.18x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 179.607 s / 180.917 s | 202.508 s | 1090 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi` | ok | 40.535 s / 41.355 s | 49.296 s | 968 MB (rss) | n/a (no oracle) | 0.23x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi-autodiff` | ok | 34.687 s / 36.014 s | 44.904 s | 988 MB (rss) | n/a (no oracle) | 0.19x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 120.23 ms / 120.24 ms | 32.422 s | 616.70 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi` | ok | 150.06 ms / 150.13 ms | 28.107 s | 616.70 MB (hbm) | n/a (no oracle) | 1.25x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi-autodiff` | ok | 152.58 ms / 152.69 ms | 31.471 s | 616.70 MB (hbm) | n/a (no oracle) | 1.27x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | ok | 3.668 s / 3.998 s | 9.757 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | ok | 102.835 s / 103.785 s | 110.808 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax-mi` | ok | 87.283 s / 90.776 s | 95.143 s | 1311 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 77.358 s / 79.990 s | 86.678 s | 1333 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | ok | 398.74 ms / 402.26 ms | 33.548 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax-mi` | ok | 511.55 ms / 514.25 ms | 33.797 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.28x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 526.29 ms / 534.06 ms | 35.146 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.32x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 6.627 s / 6.784 s | 16.552 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | ok | 117.941 s / 118.190 s | 119.862 s | 815 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi` | ok | 124.428 s / 125.057 s | 153.074 s | 1357 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi-autodiff` | ok | 109.677 s / 112.726 s | 124.638 s | 1380 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 418.95 ms / 426.76 ms | 32.404 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi` | ok | 534.55 ms / 534.83 ms | 29.951 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.28x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax-mi-autodiff` | ok | 539.08 ms / 540.71 ms | 31.569 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.29x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

