# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:15:01.131346+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 25.55 ms / 27.31 ms | 229.21 ms | 500 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 1.12 ms / 1.13 ms | 1.20 ms | 465 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 5.10 ms / 5.18 ms | 582.70 ms | 68.35 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 915.4 µs / 928.7 µs | 970.1 µs | 0.07 MB (hbm) | ✓ 0×tol | 0.18x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `nitrix-jax` | ok | 84.90 ms / 85.58 ms | 299.35 ms | 514 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.26 ms / 4.40 ms | 4.27 ms | 465 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `nitrix-jax` | ok | 3.92 ms / 3.93 ms | 657.12 ms | 74.32 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 4.44 ms / 4.55 ms | 5.47 ms | 0.13 MB (hbm) | ✓ 0×tol | 1.13x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `nitrix-jax` | ok | 432.7 µs / 445.0 µs | 165.11 ms | 514 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 56.6 µs / 58.1 µs | 103.3 µs | 465 MB (rss) | ✓ 0×tol | 0.13x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `nitrix-jax` | ok | 1.65 ms / 1.68 ms | 571.08 ms | 33.63 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 57.3 µs / 59.7 µs | 98.8 µs | 0.00 MB (hbm) | ✓ 0×tol | 0.03x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 1.240 s / 1.523 s | 1.681 s | 571 MB (rss) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 30.82 ms / 33.34 ms | 36.22 ms | 465 MB (rss) | ✓ 0×tol | 0.02x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 23.18 ms / 23.32 ms | 707.52 ms | 135.27 MB (hbm) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 29.69 ms / 30.40 ms | 35.83 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.28x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

