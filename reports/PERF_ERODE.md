# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c31df7e16dc35cf91c501118e6b204aa56220747 | bench: 5066cd0f8aa79a206a7697a2c126d51e118a6921
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T02:45:58.640685+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| erode | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `cupyx.scipy.ndimage.grey_erosion` | ok | 1.86 ms / 1.87 ms | 220.40 ms | 33.55 MB (hbm) | n/a (no oracle) | 0.01x vs nitrix-jax |
| erode | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `nitrix-jax` | ok | 351.55 ms / 351.77 ms | 1.467 s | 8724.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `cupyx.scipy.ndimage.grey_erosion` | ok | 2.31 ms / 2.32 ms | 241.54 ms | 67.11 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `nitrix-jax` | ok | 698.45 ms / 698.75 ms | 2.117 s | 16995.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `cupyx.scipy.ndimage.grey_erosion` | ok | 11.81 ms / 13.74 ms | 2.839 s | 67.11 MB (hbm) | n/a (no oracle) | — |
| erode | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `nitrix-jax` | oom | — | — | — | — | — |
| erode | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `cupyx.scipy.ndimage.grey_erosion` | ok | 2.28 ms / 2.31 ms | 179.56 ms | 67.11 MB (hbm) | n/a (no oracle) | 3.63x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `nitrix-jax` | ok | 629.2 µs / 637.7 µs | 118.44 ms | 201.33 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 4.10 ms / 4.14 ms | 71.07 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_erosion` | ok | 1.29 ms / 1.31 ms | 1.34 ms | 707 MB (rss) | ✓ 0×tol | 0.32x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleErode` | ok | 6.16 ms / 6.35 ms | 63.66 ms | 707 MB (rss) | ✓ 0×tol | 1.50x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_erosion` | ok | 135.0 µs / 145.1 µs | 176.44 ms | 0.26 MB (hbm) | ✓ 0×tol | 1.37x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 98.7 µs / 103.7 µs | 119.16 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_erosion` | ok | 1.30 ms / 1.31 ms | 1.41 ms | 0.26 MB (hbm) | ✓ 0×tol | 13.14x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleErode` | ok | 6.11 ms / 6.15 ms | 64.24 ms | 0.26 MB (hbm) | ✓ 0×tol | 61.91x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 208.2 µs / 227.6 µs | 56.79 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 1.22 ms / 1.24 ms | 1.26 ms | 707 MB (rss) | ✓ 0×tol | 5.85x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 3.33 ms / 3.43 ms | 63.07 ms | 707 MB (rss) | ✓ 0×tol | 16.01x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 120.5 µs / 129.3 µs | 302.04 ms | 0.26 MB (hbm) | ✓ 0×tol | 1.20x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 100.6 µs / 111.7 µs | 119.64 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 1.22 ms / 1.23 ms | 1.30 ms | 0.26 MB (hbm) | ✓ 0×tol | 12.09x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 3.28 ms / 3.39 ms | 72.99 ms | 0.26 MB (hbm) | ✓ 0×tol | 32.64x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 397.9 µs / 411.3 µs | 61.10 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 95.1 µs / 96.1 µs | 103.75 ms | 0.39 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 5.53 ms / 8.12 ms | 143.39 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_erosion` | ok | 1.37 ms / 1.45 ms | 1.41 ms | 707 MB (rss) | ✓ 0×tol | 0.25x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 256.4 µs / 262.9 µs | 203.12 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.67x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 381.3 µs / 412.5 µs | 805.86 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_erosion` | ok | 1.40 ms / 1.47 ms | 1.46 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.66x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 41.31 ms / 43.65 ms | 175.00 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_erosion` | ok | 6.34 ms / 6.89 ms | 6.48 ms | 707 MB (rss) | ✓ 0×tol | 0.15x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_erosion` | ok | 275.3 µs / 288.7 µs | 212.26 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.17x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 1.58 ms / 1.59 ms | 640.89 ms | 193.46 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_erosion` | ok | 6.38 ms / 6.79 ms | 6.41 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.04x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 124.65 ms / 135.08 ms | 271.51 ms | 730 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_erosion` | ok | 6.25 ms / 6.81 ms | 6.65 ms | 707 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_erosion` | ok | 288.8 µs / 300.2 µs | 250.58 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.09x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 3.09 ms / 3.13 ms | 751.47 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_erosion` | ok | 6.18 ms / 6.50 ms | 6.28 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 3.04 ms / 4.16 ms | 65.69 ms | 707 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 7.88 ms / 8.03 ms | 8.03 ms | 707 MB (rss) | ✓ 0×tol | 2.59x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 20.08 ms / 20.71 ms | 76.22 ms | 707 MB (rss) | ✓ 0×tol | 6.61x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 168.4 µs / 173.3 µs | 176.80 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.77x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 95.2 µs / 115.9 µs | 120.33 ms | 3.15 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 7.93 ms / 8.15 ms | 8.15 ms | 1.05 MB (hbm) | ✓ 0×tol | 83.35x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 20.85 ms / 22.49 ms | 79.21 ms | 1.05 MB (hbm) | ✓ 0×tol | 219.09x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

