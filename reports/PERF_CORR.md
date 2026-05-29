# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:38:39.096293+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| corr | jax-cpu | n=2000,t=1000 | `cupy.corrcoef` | skipped | — | — | — | — | — |
| corr | jax-cpu | n=2000,t=1000 | `nitrix-jax` | ok | 42.84 ms / 61.92 ms | 269.93 ms | 665 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=2000,t=1000 | `numpy.corrcoef` | ok | 70.51 ms / 100.62 ms | 98.23 ms | 575 MB (rss) | ✓ 0×tol | 1.65x vs nitrix-jax |
| corr | jax-cuda12 | n=2000,t=1000 | `cupy.corrcoef` | ok | 21.64 ms / 21.65 ms | 366.31 ms | 8.39 MB (hbm) | ✓ 2.8e-12×tol | 27.95x vs nitrix-jax |
| corr | jax-cuda12 | n=2000,t=1000 | `nitrix-jax` | ok | 774.2 µs / 787.5 µs | 553.19 ms | 111.69 MB (hbm) | ✓ 0.00082×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=2000,t=1000 | `numpy.corrcoef` | ok | 58.78 ms / 59.62 ms | 79.66 ms | 8.39 MB (hbm) | ✓ 0×tol | 75.92x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `cupy.corrcoef` | skipped | — | — | — | — | — |
| corr | jax-cpu | n=50,t=500 | `nitrix-jax` | ok | 62.1 µs / 62.9 µs | 140.32 ms | 532 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `numpy.corrcoef` | ok | 127.0 µs / 130.2 µs | 3.05 ms | 532 MB (rss) | ✓ 0×tol | 2.04x vs nitrix-jax |
| corr | jax-cuda12 | n=50,t=500 | `cupy.corrcoef` | ok | 334.0 µs / 341.0 µs | 1.185 s | 0.10 MB (hbm) | ✓ 9.3e-13×tol | 2.99x vs nitrix-jax |
| corr | jax-cuda12 | n=50,t=500 | `nitrix-jax` | ok | 111.7 µs / 113.4 µs | 374.30 ms | 71.56 MB (hbm) | ✓ 0.00022×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=50,t=500 | `numpy.corrcoef` | ok | 126.3 µs / 128.8 µs | 206.5 µs | 0.10 MB (hbm) | ✓ 0×tol | 1.13x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `cupy.corrcoef` | skipped | — | — | — | — | — |
| corr | jax-cpu | n=500,t=2000 | `nitrix-jax` | ok | 6.21 ms / 9.13 ms | 150.00 ms | 542 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `numpy.corrcoef` | ok | 7.67 ms / 10.08 ms | 14.01 ms | 532 MB (rss) | ✓ 0×tol | 1.23x vs nitrix-jax |
| corr | jax-cuda12 | n=500,t=2000 | `cupy.corrcoef` | ok | 3.71 ms / 3.73 ms | 254.04 ms | 4.19 MB (hbm) | ✓ 2.3e-12×tol | 16.05x vs nitrix-jax |
| corr | jax-cuda12 | n=500,t=2000 | `nitrix-jax` | ok | 231.4 µs / 235.1 µs | 481.82 ms | 83.50 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=500,t=2000 | `numpy.corrcoef` | ok | 8.27 ms / 30.06 ms | 87.71 ms | 4.19 MB (hbm) | ✓ 0×tol | 35.73x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

