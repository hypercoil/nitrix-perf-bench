# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 9e53019e8fa652aaa379a09ac190bb18c0d8e3a8 | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T00:53:15.201065+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 8.818 s / 9.928 s | 14.292 s | 863 MB (rss) | n/a (no oracle) | 0.72x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 12.203 s / 14.199 s | 17.405 s | 983 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 6.365 s / 6.403 s | 8.474 s | 921 MB (rss) | n/a (no oracle) | 0.52x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 295.77 ms / 298.22 ms | 36.839 s | 8783.40 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `simpleitk.demons` | ok | 11.255 s / 14.480 s | 10.627 s | 16.78 MB (hbm) | n/a (no oracle) | 38.05x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 11.801 s / 11.839 s | 14.498 s | 865 MB (rss) | n/a (no oracle) | 1.26x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 9.332 s / 9.473 s | 12.090 s | 1087 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 5.874 s / 5.996 s | 5.587 s | 957 MB (rss) | n/a (no oracle) | 0.63x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 302.67 ms / 303.47 ms | 52.805 s | 8850.51 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 5.555 s / 5.637 s | 5.615 s | 16.78 MB (hbm) | n/a (no oracle) | 18.35x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 20.286 s / 28.718 s | 25.897 s | 1214 MB (rss) | n/a (no oracle) | 0.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 23.357 s / 23.777 s | 41.487 s | 1231 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 11.234 s / 11.236 s | 11.510 s | 1287 MB (rss) | n/a (no oracle) | 0.48x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 650.22 ms / 651.05 ms | 36.789 s | 2652.21 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `simpleitk.demons` | ok | 18.968 s / 19.692 s | 19.426 s | 33.55 MB (hbm) | n/a (no oracle) | 29.17x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | ok | 1.256 s / 1.297 s | 6.979 s | 863 MB (rss) | n/a (no oracle) | 3.05x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 411.54 ms / 418.10 ms | 1.591 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 529.97 ms / 744.13 ms | 984.24 ms | 863 MB (rss) | n/a (no oracle) | 1.29x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 6.35 ms / 6.41 ms | 6.441 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 604.45 ms / 630.28 ms | 834.23 ms | 0.88 MB (hbm) | n/a (no oracle) | 95.23x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 1.350 s / 3.465 s | 11.158 s | 863 MB (rss) | n/a (no oracle) | 3.90x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 346.25 ms / 359.49 ms | 2.641 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 302.02 ms / 321.47 ms | 396.53 ms | 863 MB (rss) | n/a (no oracle) | 0.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 9.27 ms / 9.39 ms | 22.743 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 645.30 ms / 698.62 ms | 561.82 ms | 0.88 MB (hbm) | n/a (no oracle) | 69.63x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | ok | 3.101 s / 3.269 s | 12.245 s | 863 MB (rss) | n/a (no oracle) | 4.86x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 637.62 ms / 648.96 ms | 3.636 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 598.27 ms / 598.74 ms | 619.19 ms | 863 MB (rss) | n/a (no oracle) | 0.94x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 18.25 ms / 18.79 ms | 21.966 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 980.28 ms / 1.555 s | 936.37 ms | 0.88 MB (hbm) | n/a (no oracle) | 53.72x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 7.063 s / 7.233 s | 11.020 s | 863 MB (rss) | n/a (no oracle) | 1.97x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 3.585 s / 3.702 s | 7.155 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 2.340 s / 2.463 s | 2.435 s | 863 MB (rss) | n/a (no oracle) | 0.65x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 80.43 ms / 80.52 ms | 17.336 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `simpleitk.demons` | ok | 5.106 s / 6.285 s | 5.829 s | 8.39 MB (hbm) | n/a (no oracle) | 63.48x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | ok | 7.905 s / 7.938 s | 16.089 s | 863 MB (rss) | n/a (no oracle) | 2.32x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 3.401 s / 3.479 s | 7.406 s | 863 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 2.224 s / 2.349 s | 2.278 s | 863 MB (rss) | n/a (no oracle) | 0.65x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `nitrix-jax` | ok | 83.81 ms / 83.95 ms | 26.494 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20,spacing=[1, 1, 3] | `simpleitk.demons` | ok | 3.665 s / 5.073 s | 3.382 s | 8.39 MB (hbm) | n/a (no oracle) | 43.73x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

