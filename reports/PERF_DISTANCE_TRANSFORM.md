# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:42:52.404226+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform | jax-cpu | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 21.23 ms / 26.98 ms | 209.62 ms | 505 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 915.4 µs / 924.8 µs | 971.6 µs | 464 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 201.0 µs / 204.4 µs | 230.75 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 5.05 ms / 5.08 ms | 614.01 ms | 68.35 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 1.25 ms / 1.32 ms | 1.42 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.25x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `nitrix-jax` | ok | 85.14 ms / 87.74 ms | 314.63 ms | 510 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.89 ms / 3.92 ms | 4.23 ms | 464 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 147.2 µs / 150.0 µs | 967.20 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `nitrix-jax` | ok | 3.90 ms / 3.91 ms | 676.03 ms | 74.32 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.28 ms / 4.52 ms | 5.38 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.84x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32] | `nitrix-jax` | ok | 416.9 µs / 434.4 µs | 158.17 ms | 514 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 80.0 µs / 81.1 µs | 119.0 µs | 464 MB (rss) | ✓ 0×tol | 0.19x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 187.1 µs / 192.4 µs | 1.297 s | 0.00 MB (hbm) | ✓ 0×tol | 0.11x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `nitrix-jax` | ok | 1.68 ms / 1.70 ms | 560.64 ms | 33.63 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 56.5 µs / 58.0 µs | 103.6 µs | 0.00 MB (hbm) | ✓ 0×tol | 0.03x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 1.148 s / 1.166 s | 1.469 s | 569 MB (rss) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 30.56 ms / 33.93 ms | 36.11 ms | 464 MB (rss) | ✓ 0×tol | 0.03x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 222.5 µs / 224.9 µs | 298.19 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.01x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 23.32 ms / 23.44 ms | 758.39 ms | 135.27 MB (hbm) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 29.42 ms / 29.80 ms | 34.65 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.26x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

