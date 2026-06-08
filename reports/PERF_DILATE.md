# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c31df7e16dc35cf91c501118e6b204aa56220747 | bench: 5066cd0f8aa79a206a7697a2c126d51e118a6921
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T02:32:18.053632+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `cupyx.scipy.ndimage.grey_dilation` | ok | 1.87 ms / 1.89 ms | 216.07 ms | 33.55 MB (hbm) | n/a (no oracle) | 0.01x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `nitrix-jax` | ok | 351.58 ms / 351.91 ms | 1.429 s | 8724.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `cupyx.scipy.ndimage.grey_dilation` | ok | 2.35 ms / 2.37 ms | 209.48 ms | 67.11 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `nitrix-jax` | ok | 698.60 ms / 698.93 ms | 2.105 s | 16995.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `cupyx.scipy.ndimage.grey_dilation` | ok | 12.02 ms / 13.63 ms | 21.361 s | 67.11 MB (hbm) | n/a (no oracle) | — |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `nitrix-jax` | oom | — | — | — | — | — |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `cupyx.scipy.ndimage.grey_dilation` | ok | 2.29 ms / 2.31 ms | 172.21 ms | 67.11 MB (hbm) | n/a (no oracle) | 3.62x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `nitrix-jax` | ok | 632.4 µs / 637.9 µs | 123.82 ms | 201.33 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 4.08 ms / 4.50 ms | 65.57 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_dilation` | ok | 1.31 ms / 1.33 ms | 1.35 ms | 706 MB (rss) | ✓ 0×tol | 0.32x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleDilate` | ok | 6.09 ms / 6.18 ms | 61.53 ms | 706 MB (rss) | ✓ 0×tol | 1.49x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `cupyx.scipy.ndimage.grey_dilation` | ok | 137.4 µs / 205.0 µs | 167.11 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.94x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `nitrix-jax` | ok | 145.8 µs / 191.2 µs | 138.01 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `scipy.ndimage.grey_dilation` | ok | 1.29 ms / 1.31 ms | 1.34 ms | 0.26 MB (hbm) | ✓ 0×tol | 8.88x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=15 | `simpleitk.GrayscaleDilate` | ok | 6.06 ms / 6.12 ms | 60.57 ms | 0.26 MB (hbm) | ✓ 0×tol | 41.59x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 336.9 µs / 358.8 µs | 67.10 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 1.21 ms / 1.22 ms | 1.28 ms | 706 MB (rss) | ✓ 0×tol | 3.60x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 3.30 ms / 3.45 ms | 66.37 ms | 706 MB (rss) | ✓ 0×tol | 9.81x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 126.8 µs / 131.7 µs | 357.76 ms | 0.26 MB (hbm) | ✓ 0×tol | 1.39x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 91.1 µs / 93.8 µs | 119.64 ms | 0.79 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 1.21 ms / 1.22 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 13.26x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 4.87 ms / 5.42 ms | 143.75 ms | 0.26 MB (hbm) | ✓ 0×tol | 53.43x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 406.3 µs / 415.7 µs | 66.83 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=box,size=3,dtype=float16 | `nitrix-jax` | ok | 93.8 µs / 106.1 µs | 108.69 ms | 0.39 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 6.88 ms / 10.27 ms | 135.63 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_dilation` | ok | 1.34 ms / 1.38 ms | 2.74 ms | 706 MB (rss) | ✓ 0×tol | 0.20x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 292.5 µs / 299.4 µs | 227.96 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.78x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 375.5 µs / 388.8 µs | 814.85 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_dilation` | ok | 1.34 ms / 1.37 ms | 1.39 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.56x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 39.56 ms / 46.46 ms | 161.51 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_dilation` | ok | 6.30 ms / 6.50 ms | 6.39 ms | 706 MB (rss) | ✓ 0×tol | 0.16x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `cupyx.scipy.ndimage.grey_dilation` | ok | 307.3 µs / 320.1 µs | 208.18 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.19x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `nitrix-jax` | ok | 1.58 ms / 1.64 ms | 663.64 ms | 193.46 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],se=disk,radius=7 | `scipy.ndimage.grey_dilation` | ok | 6.29 ms / 6.39 ms | 6.32 ms | 0.26 MB (hbm) | ✓ 0×tol | 3.99x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 125.05 ms / 141.70 ms | 270.83 ms | 729 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_dilation` | ok | 5.98 ms / 6.38 ms | 6.25 ms | 706 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_dilation` | ok | 344.5 µs / 408.9 µs | 277.26 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.11x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 3.09 ms / 3.23 ms | 676.09 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_dilation` | ok | 5.89 ms / 5.93 ms | 6.13 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.91x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 3.25 ms / 3.80 ms | 73.81 ms | 706 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 7.88 ms / 7.95 ms | 8.03 ms | 706 MB (rss) | ✓ 0×tol | 2.43x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 20.22 ms / 20.76 ms | 78.40 ms | 706 MB (rss) | ✓ 0×tol | 6.22x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 180.5 µs / 190.2 µs | 170.50 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.67x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 108.0 µs / 116.7 µs | 115.77 ms | 3.15 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_dilation` | ok | 7.86 ms / 7.91 ms | 8.03 ms | 1.05 MB (hbm) | ✓ 0×tol | 72.75x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `simpleitk.GrayscaleDilate` | ok | 20.19 ms / 20.96 ms | 77.76 ms | 1.05 MB (hbm) | ✓ 0×tol | 186.88x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

