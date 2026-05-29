# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:43:50.095053+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| spatial_transform | jax-cpu | shape=[256, 256] | `cupyx.scipy.ndimage.map_coordinates` | skipped | — | — | — | — | — |
| spatial_transform | jax-cpu | shape=[256, 256] | `nitrix-jax` | ok | 376.7 µs / 447.3 µs | 137.27 ms | 493 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[256, 256] | `scipy.ndimage.map_coordinates` | ok | 1.86 ms / 1.87 ms | 3.54 ms | 453 MB (rss) | ✓ 5.5e-05×tol | 4.93x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[256, 256] | `cupyx.scipy.ndimage.map_coordinates` | ok | 52.7 µs / 55.4 µs | 328.89 ms | 0.79 MB (hbm) | ✓ 0.00075×tol | 0.56x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[256, 256] | `nitrix-jax` | ok | 94.5 µs / 98.5 µs | 180.28 ms | 1.31 MB (hbm) | ✓ 0.00098×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[256, 256] | `scipy.ndimage.map_coordinates` | ok | 1.95 ms / 1.97 ms | 1.98 ms | 0.79 MB (hbm) | ✓ 5.5e-05×tol | 20.66x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `cupyx.scipy.ndimage.map_coordinates` | skipped | — | — | — | — | — |
| spatial_transform | jax-cpu | shape=[64, 64] | `nitrix-jax` | ok | 62.6 µs / 69.9 µs | 131.48 ms | 493 MB (rss) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `scipy.ndimage.map_coordinates` | ok | 120.6 µs / 122.3 µs | 142.7 µs | 453 MB (rss) | ✓ 5.2e-05×tol | 1.93x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[64, 64] | `cupyx.scipy.ndimage.map_coordinates` | ok | 53.7 µs / 60.3 µs | 516.37 ms | 0.05 MB (hbm) | ✓ 0.00051×tol | 0.56x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[64, 64] | `nitrix-jax` | ok | 96.4 µs / 99.0 µs | 181.00 ms | 0.08 MB (hbm) | ✓ 0.00043×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cuda12 | shape=[64, 64] | `scipy.ndimage.map_coordinates` | ok | 126.5 µs / 127.9 µs | 154.1 µs | 0.05 MB (hbm) | ✓ 5.2e-05×tol | 1.31x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

