# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:14:06.750495+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `nitrix-jax` | ok | 3.20 ms / 3.71 ms | 165.96 ms | 527 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 731.9 µs / 930.5 µs | 1.07 ms | 455 MB (rss) | ✓ 0.00021×tol | 0.23x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[256, 256],sigma=1.5 | `nitrix-jax` | ok | 129.9 µs / 133.1 µs | 540.93 ms | 69.21 MB (hbm) | ✓ 0.00072×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[256, 256],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 702.2 µs / 707.3 µs | 1.08 ms | 0.26 MB (hbm) | ✓ 0.00021×tol | 5.41x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `nitrix-jax` | ok | 17.83 ms / 19.85 ms | 184.25 ms | 597 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 4.68 ms / 4.75 ms | 4.78 ms | 455 MB (rss) | ✓ 0.00016×tol | 0.26x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64, 64],sigma=1.5 | `nitrix-jax` | ok | 162.9 µs / 168.8 µs | 677.59 ms | 615.65 MB (hbm) | ✓ 0.00055×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 4.65 ms / 4.71 ms | 4.76 ms | 1.05 MB (hbm) | ✓ 0.00016×tol | 28.52x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `nitrix-jax` | ok | 367.5 µs / 378.8 µs | 123.29 ms | 506 MB (rss) | ✓ 0.00047×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 90.3 µs / 93.1 µs | 143.9 µs | 455 MB (rss) | ✓ 0.00019×tol | 0.25x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64],sigma=1.5 | `nitrix-jax` | ok | 121.1 µs / 126.0 µs | 1.306 s | 75.81 MB (hbm) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 96.8 µs / 100.1 µs | 135.2 µs | 0.02 MB (hbm) | ✓ 0.00019×tol | 0.80x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

