# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c54bc81807fd0b81c371b5904a98e8e6f3d88a93 | bench: 7be151160d256117f2a68003be1befe98a76a202
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T17:49:44.043275+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | ok | 4.294 s / 4.521 s | 11.932 s | 863 MB (rss) | n/a (no oracle) | 0.62x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | ok | 39.421 s / 39.476 s | 47.870 s | 863 MB (rss) | n/a (no oracle) | 5.73x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 6.875 s / 6.888 s | 10.558 s | 870 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | data=mni152,resolution=2,levels=2,iters=20 | `simpleitk.demons` | ok | 2.719 s / 2.721 s | 2.783 s | 863 MB (rss) | n/a (no oracle) | 0.40x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `nitrix-jax` | ok | 117.75 ms / 117.93 ms | 18.191 s | 15019.31 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=20 | `simpleitk.demons` | ok | 4.189 s / 4.426 s | 4.260 s | 16.78 MB (hbm) | n/a (no oracle) | 35.58x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 8.271 s / 10.555 s | 14.108 s | 863 MB (rss) | n/a (no oracle) | 0.85x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | ok | 125.564 s / 125.849 s | 126.584 s | 923 MB (rss) | n/a (no oracle) | 12.91x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 9.726 s / 9.882 s | 12.021 s | 952 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 5.642 s / 6.109 s | 5.816 s | 931 MB (rss) | n/a (no oracle) | 0.58x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 295.31 ms / 295.56 ms | 30.901 s | 8783.40 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 8.630 s / 11.342 s | 10.291 s | 16.78 MB (hbm) | n/a (no oracle) | 29.22x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 11.573 s / 11.655 s | 14.499 s | 863 MB (rss) | n/a (no oracle) | 1.17x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | ok | 84.887 s / 84.913 s | 85.358 s | 977 MB (rss) | n/a (no oracle) | 8.55x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 9.930 s / 9.937 s | 12.774 s | 1089 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 5.689 s / 6.114 s | 5.833 s | 934 MB (rss) | n/a (no oracle) | 0.57x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 301.42 ms / 302.69 ms | 52.029 s | 8850.51 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 9.396 s / 9.732 s | 10.122 s | 16.78 MB (hbm) | n/a (no oracle) | 31.17x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 21.734 s / 28.471 s | 22.387 s | 1213 MB (rss) | n/a (no oracle) | 1.15x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | timeout | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 18.856 s / 18.962 s | 30.145 s | 1227 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 11.453 s / 11.917 s | 11.583 s | 1274 MB (rss) | n/a (no oracle) | 0.61x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 647.53 ms / 650.56 ms | 30.257 s | 2652.21 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 19.916 s / 20.057 s | 19.723 s | 33.55 MB (hbm) | n/a (no oracle) | 30.76x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | ok | 805.55 ms / 851.82 ms | 5.362 s | 863 MB (rss) | n/a (no oracle) | 3.01x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | ok | 2.641 s / 2.652 s | 3.399 s | 863 MB (rss) | n/a (no oracle) | 9.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 267.57 ms / 280.57 ms | 1.177 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 287.93 ms / 291.64 ms | 1.164 s | 863 MB (rss) | n/a (no oracle) | 1.08x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 6.37 ms / 6.38 ms | 3.714 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 373.49 ms / 613.45 ms | 659.32 ms | 0.88 MB (hbm) | n/a (no oracle) | 58.67x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 684.54 ms / 700.24 ms | 4.133 s | 863 MB (rss) | n/a (no oracle) | 1.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 3.829 s / 3.830 s | 4.405 s | 863 MB (rss) | n/a (no oracle) | 10.46x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 365.99 ms / 368.28 ms | 3.023 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 291.18 ms / 296.16 ms | 361.65 ms | 863 MB (rss) | n/a (no oracle) | 0.80x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 9.53 ms / 9.59 ms | 7.761 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 689.78 ms / 954.14 ms | 606.06 ms | 0.88 MB (hbm) | n/a (no oracle) | 72.37x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | ok | 812.71 ms / 860.77 ms | 6.182 s | 863 MB (rss) | n/a (no oracle) | 1.28x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | ok | 8.206 s / 8.215 s | 8.970 s | 863 MB (rss) | n/a (no oracle) | 12.91x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 635.80 ms / 652.77 ms | 3.663 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 578.76 ms / 616.70 ms | 627.78 ms | 863 MB (rss) | n/a (no oracle) | 0.91x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 18.43 ms / 18.45 ms | 8.690 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 1.045 s / 1.507 s | 936.29 ms | 0.88 MB (hbm) | n/a (no oracle) | 56.67x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 4.498 s / 5.712 s | 9.660 s | 863 MB (rss) | n/a (no oracle) | 1.28x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | ok | 35.338 s / 35.343 s | 36.077 s | 863 MB (rss) | n/a (no oracle) | 10.09x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 3.504 s / 3.505 s | 7.148 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 2.232 s / 2.462 s | 2.299 s | 863 MB (rss) | n/a (no oracle) | 0.64x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 79.46 ms / 80.11 ms | 9.907 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 4.032 s / 4.243 s | 4.023 s | 8.39 MB (hbm) | n/a (no oracle) | 50.75x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 8.428 s / 9.245 s | 12.449 s | 863 MB (rss) | n/a (no oracle) | 2.44x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | ok | 35.811 s / 35.868 s | 37.560 s | 863 MB (rss) | n/a (no oracle) | 10.36x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 3.456 s / 3.485 s | 7.029 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 2.254 s / 2.427 s | 2.288 s | 863 MB (rss) | n/a (no oracle) | 0.65x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 84.16 ms / 84.53 ms | 12.084 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 4.195 s / 4.294 s | 4.702 s | 8.39 MB (hbm) | n/a (no oracle) | 49.84x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

