# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 00e7282111c5ee516813bbf6f41e4616cbaff125 | bench: 3e097324252de4a21f5cfae21464658bc4f599c4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T20:41:50.084744+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_erosion` | ok | 144.0 µs / 159.5 µs | 4.472 s | 0.26 MB (hbm) | ✓ 0×tol | 1.43x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 100.8 µs / 105.4 µs | 116.90 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_erosion` | ok | 1.30 ms / 1.31 ms | 1.34 ms | 0.26 MB (hbm) | ✓ 0×tol | 12.87x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleErode` | ok | 6.14 ms / 6.19 ms | 61.15 ms | 0.26 MB (hbm) | ✓ 0×tol | 60.85x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 119.7 µs / 125.8 µs | 4.654 s | 0.26 MB (hbm) | ✓ 0×tol | 1.16x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 102.9 µs / 105.0 µs | 108.59 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 1.22 ms / 1.23 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 11.82x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 3.28 ms / 3.31 ms | 63.12 ms | 0.26 MB (hbm) | ✓ 0×tol | 31.89x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 91.5 µs / 92.7 µs | 101.22 ms | 0.39 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 257.4 µs / 268.5 µs | 2.343 s | 0.26 MB (hbm) | ✓ 0×tol | 0.67x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 384.1 µs / 388.2 µs | 686.45 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_erosion` | ok | 1.37 ms / 1.39 ms | 1.44 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.57x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_erosion` | ok | 277.9 µs / 285.0 µs | 2.315 s | 0.26 MB (hbm) | ✓ 0×tol | 0.18x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 1.57 ms / 1.58 ms | 623.53 ms | 193.46 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_erosion` | ok | 6.37 ms / 6.82 ms | 6.33 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.05x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_erosion` | ok | 280.6 µs / 292.8 µs | 2.413 s | 1.05 MB (hbm) | ✓ 0×tol | 0.09x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 3.05 ms / 3.07 ms | 705.71 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_erosion` | ok | 6.08 ms / 6.20 ms | 6.87 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.99x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 166.3 µs / 176.6 µs | 6.469 s | 1.05 MB (hbm) | ✓ 0×tol | 1.46x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 113.8 µs / 122.6 µs | 121.42 ms | 3.15 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_erosion` | ok | 7.85 ms / 8.12 ms | 8.38 ms | 1.05 MB (hbm) | ✓ 0×tol | 68.98x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleErode` | ok | 20.15 ms / 20.22 ms | 79.03 ms | 1.05 MB (hbm) | ✓ 0×tol | 177.10x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

