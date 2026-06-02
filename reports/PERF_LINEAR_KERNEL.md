# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 7159b33d2e4fe9f95c9156aa9ceffdafd949591d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-01T23:49:31.074077+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| linear_kernel | jax-cpu | n=2048,d=64 | `cupy.linear_kernel` | skipped | — | — | — | — | — |
| linear_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 3.16 ms / 6.90 ms | 120.10 ms | 671 MB (rss) | ✓ 0.035×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cpu | n=2048,d=64 | `sklearn.linear_kernel` | ok | 11.80 ms / 15.24 ms | 17.23 ms | 671 MB (rss) | ✓ 0.035×tol | 3.74x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=2048,d=64 | `cupy.linear_kernel` | ok | 86.9 µs / 88.0 µs | 64.96 ms | 0.52 MB (hbm) | ✓ 0.035×tol | 0.52x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 167.7 µs / 176.8 µs | 228.53 ms | 105.38 MB (hbm) | ✓ 0.035×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.linear_kernel` | ok | 20.47 ms / 41.75 ms | 44.14 ms | 0.52 MB (hbm) | ✓ 0.035×tol | 122.05x vs nitrix-jax |
| linear_kernel | jax-cpu | n=4096,d=64 | `cupy.linear_kernel` | skipped | — | — | — | — | — |
| linear_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 37.01 ms / 45.30 ms | 156.49 ms | 851 MB (rss) | ✓ 0.048×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cpu | n=4096,d=64 | `sklearn.linear_kernel` | ok | 69.53 ms / 85.98 ms | 84.95 ms | 831 MB (rss) | ✓ 0.048×tol | 1.88x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=4096,d=64 | `cupy.linear_kernel` | ok | 286.1 µs / 293.3 µs | 63.12 ms | 1.05 MB (hbm) | ✓ 0.048×tol | 0.72x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 397.8 µs / 407.0 µs | 444.29 ms | 206.57 MB (hbm) | ✓ 0.048×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.linear_kernel` | ok | 68.43 ms / 69.31 ms | 66.06 ms | 1.05 MB (hbm) | ✓ 0.048×tol | 172.00x vs nitrix-jax |
| linear_kernel | jax-cpu | n=512,d=64 | `cupy.linear_kernel` | skipped | — | — | — | — | — |
| linear_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 293.4 µs / 335.5 µs | 86.75 ms | 671 MB (rss) | ✓ 0.026×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cpu | n=512,d=64 | `sklearn.linear_kernel` | ok | 562.7 µs / 573.5 µs | 1.04 ms | 671 MB (rss) | ✓ 0.026×tol | 1.92x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=512,d=64 | `cupy.linear_kernel` | ok | 27.5 µs / 28.5 µs | 63.98 ms | 0.13 MB (hbm) | ✓ 0.026×tol | 0.25x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 111.7 µs / 114.0 µs | 181.78 ms | 74.45 MB (hbm) | ✓ 0.026×tol | 1.00x vs nitrix-jax |
| linear_kernel | jax-cuda12 | n=512,d=64 | `sklearn.linear_kernel` | ok | 560.1 µs / 567.1 µs | 1.24 ms | 0.13 MB (hbm) | ✓ 0.026×tol | 5.01x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

