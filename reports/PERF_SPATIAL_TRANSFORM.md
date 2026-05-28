# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:15:49.739759+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| spatial_transform | jax-cpu | shape=[256, 256] | `nitrix-jax` | ok | 462.1 µs / 504.0 µs | 141.68 ms | 498 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[256, 256] | `scipy.ndimage.map_coordinates` | ok | 1.85 ms / 1.92 ms | 2.04 ms | 453 MB (rss) | ✓ 5.5e-05×tol | 4.01x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[256, 256] | `nitrix-jax` | ok | 98.8 µs / 111.5 µs | 174.02 ms | 1.31 MB (hbm) | ✓ 0.00098×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[256, 256] | `scipy.ndimage.map_coordinates` | ok | 1.94 ms / 1.98 ms | 2.02 ms | 0.79 MB (hbm) | ✓ 5.5e-05×tol | 19.66x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `nitrix-jax` | ok | 75.3 µs / 86.2 µs | 133.27 ms | 494 MB (rss) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `scipy.ndimage.map_coordinates` | ok | 122.3 µs / 122.6 µs | 143.3 µs | 453 MB (rss) | ✓ 5.2e-05×tol | 1.62x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[64, 64] | `nitrix-jax` | ok | 96.4 µs / 98.3 µs | 175.57 ms | 0.08 MB (hbm) | ✓ 0.00043×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[64, 64] | `scipy.ndimage.map_coordinates` | ok | 126.8 µs / 127.8 µs | 146.1 µs | 0.05 MB (hbm) | ✓ 5.2e-05×tol | 1.32x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

