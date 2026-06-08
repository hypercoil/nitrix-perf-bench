# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c31df7e16dc35cf91c501118e6b204aa56220747 | bench: 5066cd0f8aa79a206a7697a2c126d51e118a6921
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T02:51:17.873342+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| open | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `cupyx.scipy.ndimage.grey_opening` | ok | 3.29 ms / 3.33 ms | 212.98 ms | 33.55 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[128, 128, 128],se=ball,radius=2,batch=4,tier=large | `nitrix-jax` | ok | 702.79 ms / 703.06 ms | 2.127 s | 8724.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `cupyx.scipy.ndimage.grey_opening` | ok | 4.49 ms / 4.57 ms | 266.81 ms | 67.11 MB (hbm) | n/a (no oracle) | 0.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=2,tier=large | `nitrix-jax` | ok | 1.396 s / 1.397 s | 3.345 s | 16995.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `cupyx.scipy.ndimage.grey_opening` | ok | 27.08 ms / 27.19 ms | 278.09 ms | 67.11 MB (hbm) | n/a (no oracle) | — |
| open | jax-cuda12 | shape=[256, 256, 256],se=ball,radius=4,tier=large | `nitrix-jax` | oom | — | — | — | — | — |
| open | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `cupyx.scipy.ndimage.grey_opening` | ok | 4.50 ms / 4.99 ms | 174.34 ms | 67.11 MB (hbm) | n/a (no oracle) | 3.91x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256, 256],se=box,size=3,tier=large | `nitrix-jax` | ok | 1.15 ms / 1.17 ms | 153.71 ms | 268.44 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| open | jax-cpu | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | skipped | — | — | — | — | — |
| open | jax-cpu | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 397.0 µs / 422.3 µs | 65.48 ms | 708 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cpu | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 2.21 ms / 2.25 ms | 2.25 ms | 708 MB (rss) | ✓ 0×tol | 5.58x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 221.2 µs / 229.9 µs | 237.44 ms | 0.26 MB (hbm) | ✓ 0×tol | 2.23x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 99.3 µs / 101.2 µs | 124.93 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 2.23 ms / 2.25 ms | 2.28 ms | 0.26 MB (hbm) | ✓ 0×tol | 22.40x vs nitrix-jax |
| open | jax-cpu | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_opening` | skipped | — | — | — | — | — |
| open | jax-cpu | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 12.33 ms / 17.53 ms | 184.96 ms | 708 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cpu | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_opening` | ok | 2.73 ms / 2.89 ms | 2.81 ms | 708 MB (rss) | ✓ 0×tol | 0.22x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 524.4 µs / 531.6 µs | 213.99 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.82x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 638.2 µs / 660.1 µs | 897.95 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_opening` | ok | 2.75 ms / 3.09 ms | 4.14 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.30x vs nitrix-jax |
| open | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_opening` | skipped | — | — | — | — | — |
| open | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 214.66 ms / 226.11 ms | 383.14 ms | 739 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cpu | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_opening` | ok | 12.13 ms / 12.31 ms | 27.81 ms | 708 MB (rss) | ✓ 0×tol | 0.06x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_opening` | ok | 563.0 µs / 582.4 µs | 200.98 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.09x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 6.00 ms / 6.11 ms | 870.53 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_opening` | ok | 12.09 ms / 12.31 ms | 12.14 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.02x vs nitrix-jax |
| open | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | skipped | — | — | — | — | — |
| open | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 4.04 ms / 4.43 ms | 79.15 ms | 708 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cpu | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 14.69 ms / 14.82 ms | 14.85 ms | 708 MB (rss) | ✓ 0×tol | 3.63x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 328.7 µs / 349.4 µs | 173.72 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.75x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 119.4 µs / 121.1 µs | 148.79 ms | 4.19 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 18.65 ms / 19.49 ms | 17.97 ms | 1.05 MB (hbm) | ✓ 0×tol | 156.19x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

