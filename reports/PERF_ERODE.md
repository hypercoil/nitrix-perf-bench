# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:14:28.106012+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| erode | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 723.9 µs / 822.8 µs | 141.37 ms | 513 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.21 ms / 1.23 ms | 1.33 ms | 449 MB (rss) | ✓ 0×tol | 1.68x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 147.3 µs / 156.1 µs | 663.33 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.21 ms / 1.22 ms | 1.24 ms | 0.26 MB (hbm) | ✓ 0×tol | 8.22x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 57.7 µs / 67.4 µs | 122.76 ms | 508 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 52.3 µs / 54.0 µs | 101.0 µs | 449 MB (rss) | ✓ 0×tol | 0.91x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 140.7 µs / 143.8 µs | 3.343 s | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 59.1 µs / 73.0 µs | 117.6 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.42x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

