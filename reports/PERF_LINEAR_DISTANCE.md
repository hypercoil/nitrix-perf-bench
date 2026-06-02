# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 7159b33d2e4fe9f95c9156aa9ceffdafd949591d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-01T23:50:18.007966+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| linear_distance | jax-cpu | n=2048,d=64 | `cupy.linear_distance` | skipped | — | — | — | — | — |
| linear_distance | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 4.35 ms / 10.40 ms | 164.16 ms | 708 MB (rss) | ✓ 0.61×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cpu | n=2048,d=64 | `sklearn.euclidean_distances` | ok | 21.46 ms / 36.44 ms | 40.83 ms | 667 MB (rss) | ✓ 6e-05×tol | 4.93x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=2048,d=64 | `cupy.linear_distance` | ok | 476.5 µs / 490.9 µs | 109.81 ms | 0.52 MB (hbm) | ✓ 0.00066×tol | 2.58x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 184.4 µs / 190.0 µs | 419.44 ms | 105.38 MB (hbm) | ✓ 0.61×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=2048,d=64 | `sklearn.euclidean_distances` | ok | 30.45 ms / 54.58 ms | 90.17 ms | 0.52 MB (hbm) | ✓ 6e-05×tol | 165.12x vs nitrix-jax |
| linear_distance | jax-cpu | n=4096,d=64 | `cupy.linear_distance` | skipped | — | — | — | — | — |
| linear_distance | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 32.11 ms / 42.18 ms | 415.16 ms | 883 MB (rss) | ✓ 0.61×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cpu | n=4096,d=64 | `sklearn.euclidean_distances` | ok | 169.55 ms / 202.62 ms | 179.42 ms | 821 MB (rss) | ✓ 6e-05×tol | 5.28x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=4096,d=64 | `cupy.linear_distance` | ok | 2.72 ms / 2.75 ms | 123.57 ms | 1.05 MB (hbm) | ✓ 0.00068×tol | 2.86x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 953.3 µs / 967.9 µs | 757.37 ms | 206.57 MB (hbm) | ✓ 0.61×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=4096,d=64 | `sklearn.euclidean_distances` | ok | 134.56 ms / 141.56 ms | 141.99 ms | 1.05 MB (hbm) | ✓ 6e-05×tol | 141.15x vs nitrix-jax |
| linear_distance | jax-cpu | n=512,d=64 | `cupy.linear_distance` | skipped | — | — | — | — | — |
| linear_distance | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 295.8 µs / 312.6 µs | 135.11 ms | 667 MB (rss) | ✓ 0.46×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cpu | n=512,d=64 | `sklearn.euclidean_distances` | ok | 1.58 ms / 2.18 ms | 3.47 ms | 667 MB (rss) | ✓ 5.9e-05×tol | 5.33x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=512,d=64 | `cupy.linear_distance` | ok | 100.9 µs / 104.1 µs | 112.50 ms | 0.13 MB (hbm) | ✓ 0.00052×tol | 0.89x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 112.9 µs / 114.0 µs | 360.93 ms | 74.45 MB (hbm) | ✓ 0.46×tol | 1.00x vs nitrix-jax |
| linear_distance | jax-cuda12 | n=512,d=64 | `sklearn.euclidean_distances` | ok | 1.63 ms / 2.72 ms | 11.43 ms | 0.13 MB (hbm) | ✓ 5.9e-05×tol | 14.48x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

