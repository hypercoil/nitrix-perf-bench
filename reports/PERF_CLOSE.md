# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c31df7e16dc35cf91c501118e6b204aa56220747 | bench: 5066cd0f8aa79a206a7697a2c126d51e118a6921
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T02:55:56.871636+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| close | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `cupyx.scipy.ndimage.grey_closing` | ok | 3.29 ms / 3.35 ms | 209.19 ms | 33.55 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `nitrix-jax` | ok | 702.73 ms / 702.91 ms | 2.109 s | 8724.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `cupyx.scipy.ndimage.grey_closing` | ok | 4.59 ms / 4.84 ms | 254.70 ms | 67.11 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `nitrix-jax` | ok | 1.394 s / 1.395 s | 3.563 s | 16995.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `cupyx.scipy.ndimage.grey_closing` | ok | 26.81 ms / 27.01 ms | 276.24 ms | 67.11 MB (hbm) | n/a (no oracle) | — |
| close | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `nitrix-jax` | oom | — | — | — | — | — |
| close | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `cupyx.scipy.ndimage.grey_closing` | ok | 4.50 ms / 5.05 ms | 180.45 ms | 67.11 MB (hbm) | n/a (no oracle) | 3.78x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `nitrix-jax` | ok | 1.19 ms / 1.21 ms | 170.45 ms | 268.44 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| close | jax-cpu | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | skipped | — | — | — | — | — |
| close | jax-cpu | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 678.0 µs / 821.3 µs | 72.49 ms | 709 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cpu | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 2.22 ms / 2.24 ms | 2.34 ms | 709 MB (rss) | ✓ 0×tol | 3.27x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 235.0 µs / 247.4 µs | 215.96 ms | 0.26 MB (hbm) | ✓ 0×tol | 2.49x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 94.5 µs / 99.7 µs | 124.02 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 2.21 ms / 2.25 ms | 2.26 ms | 0.26 MB (hbm) | ✓ 0×tol | 23.37x vs nitrix-jax |
| close | jax-cpu | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_closing` | skipped | — | — | — | — | — |
| close | jax-cpu | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 12.48 ms / 19.56 ms | 192.03 ms | 709 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cpu | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_closing` | ok | 2.76 ms / 2.96 ms | 2.79 ms | 709 MB (rss) | ✓ 0×tol | 0.22x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 526.3 µs / 537.0 µs | 211.63 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.83x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 636.6 µs / 644.6 µs | 873.22 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_closing` | ok | 2.71 ms / 2.85 ms | 3.05 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.26x vs nitrix-jax |
| close | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_closing` | skipped | — | — | — | — | — |
| close | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 221.77 ms / 236.72 ms | 454.58 ms | 739 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_closing` | ok | 12.11 ms / 13.34 ms | 12.73 ms | 709 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_closing` | ok | 706.1 µs / 739.7 µs | 242.51 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.12x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 6.03 ms / 6.14 ms | 836.76 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_closing` | ok | 12.09 ms / 12.29 ms | 12.63 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.00x vs nitrix-jax |
| close | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | skipped | — | — | — | — | — |
| close | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 3.97 ms / 4.93 ms | 74.77 ms | 709 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 14.72 ms / 14.90 ms | 15.15 ms | 709 MB (rss) | ✓ 0×tol | 3.70x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 322.7 µs / 338.7 µs | 169.56 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.55x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 126.4 µs / 141.9 µs | 144.06 ms | 4.19 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 14.91 ms / 14.93 ms | 15.02 ms | 1.05 MB (hbm) | ✓ 0×tol | 117.99x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

