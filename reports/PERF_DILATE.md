# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 00e7282111c5ee516813bbf6f41e4616cbaff125 | bench: 3e097324252de4a21f5cfae21464658bc4f599c4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T20:38:13.473707+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_dilation` | ok | 129.8 µs / 133.9 µs | 4.772 s | 0.26 MB (hbm) | ✓ 0×tol | 1.27x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 102.3 µs / 111.0 µs | 119.55 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_dilation` | ok | 1.31 ms / 1.32 ms | 1.35 ms | 0.26 MB (hbm) | ✓ 0×tol | 12.80x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleDilate` | ok | 6.02 ms / 6.09 ms | 60.20 ms | 0.26 MB (hbm) | ✓ 0×tol | 58.81x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 125.1 µs / 128.6 µs | 6.793 s | 0.26 MB (hbm) | ✓ 0×tol | 1.31x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 95.5 µs / 96.9 µs | 353.99 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 1.21 ms / 1.22 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 12.69x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 3.31 ms / 3.51 ms | 64.30 ms | 0.26 MB (hbm) | ✓ 0×tol | 34.63x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 98.8 µs / 102.0 µs | 102.84 ms | 0.39 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 300.7 µs / 307.7 µs | 13.547 s | 0.26 MB (hbm) | ✓ 0×tol | 0.82x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 368.1 µs / 375.9 µs | 1.368 s | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_dilation` | ok | 1.34 ms / 1.38 ms | 1.38 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.63x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_dilation` | ok | 308.7 µs / 314.7 µs | 19.474 s | 0.26 MB (hbm) | ✓ 0×tol | 0.20x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 1.58 ms / 1.63 ms | 704.77 ms | 193.46 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_dilation` | ok | 6.25 ms / 6.36 ms | 6.29 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.96x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_dilation` | ok | 324.5 µs / 331.1 µs | 2.579 s | 1.05 MB (hbm) | ✓ 0×tol | 0.11x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 3.07 ms / 3.15 ms | 729.13 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_dilation` | ok | 5.89 ms / 5.95 ms | 6.54 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.92x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 177.9 µs / 184.7 µs | 6.413 s | 1.05 MB (hbm) | ✓ 0×tol | 1.78x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 100.1 µs / 110.7 µs | 114.62 ms | 3.15 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 7.85 ms / 7.90 ms | 7.92 ms | 1.05 MB (hbm) | ✓ 0×tol | 78.40x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 20.08 ms / 24.34 ms | 87.33 ms | 1.05 MB (hbm) | ✓ 0×tol | 200.50x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

