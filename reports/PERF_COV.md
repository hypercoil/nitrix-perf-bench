# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:38:05.732707+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cov | jax-cpu | c=2000,n_obs=1000 | `cupy.cov` | skipped | — | — | — | — | — |
| cov | jax-cpu | c=2000,n_obs=1000 | `nitrix-jax` | ok | 58.69 ms / 74.00 ms | 184.29 ms | 674 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=2000,n_obs=1000 | `numpy.cov` | ok | 69.54 ms / 82.65 ms | 120.17 ms | 575 MB (rss) | ✓ 0×tol | 1.18x vs nitrix-jax |
| cov | jax-cuda12 | c=2000,n_obs=1000 | `cupy.cov` | ok | 21.27 ms / 21.28 ms | 274.62 ms | 8.39 MB (hbm) | ✓ 2.9e-12×tol | 29.09x vs nitrix-jax |
| cov | jax-cuda12 | c=2000,n_obs=1000 | `nitrix-jax` | ok | 730.9 µs / 738.2 µs | 492.32 ms | 111.69 MB (hbm) | ✓ 0.00081×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=2000,n_obs=1000 | `numpy.cov` | ok | 52.31 ms / 52.79 ms | 72.87 ms | 8.39 MB (hbm) | ✓ 0×tol | 71.57x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `cupy.cov` | skipped | — | — | — | — | — |
| cov | jax-cpu | c=50,n_obs=500 | `nitrix-jax` | ok | 79.1 µs / 80.6 µs | 130.60 ms | 532 MB (rss) | ✓ 0.00053×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `numpy.cov` | ok | 104.6 µs / 108.6 µs | 59.45 ms | 532 MB (rss) | ✓ 0×tol | 1.32x vs nitrix-jax |
| cov | jax-cuda12 | c=50,n_obs=500 | `cupy.cov` | ok | 258.3 µs / 270.2 µs | 541.86 ms | 0.10 MB (hbm) | ✓ 9e-13×tol | 2.33x vs nitrix-jax |
| cov | jax-cuda12 | c=50,n_obs=500 | `nitrix-jax` | ok | 111.0 µs / 116.3 µs | 381.01 ms | 71.56 MB (hbm) | ✓ 0.00015×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=50,n_obs=500 | `numpy.cov` | ok | 106.1 µs / 107.9 µs | 214.9 µs | 0.10 MB (hbm) | ✓ 0×tol | 0.96x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `cupy.cov` | skipped | — | — | — | — | — |
| cov | jax-cpu | c=500,n_obs=2000 | `nitrix-jax` | ok | 5.94 ms / 6.17 ms | 164.17 ms | 549 MB (rss) | ✓ 0.00076×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `numpy.cov` | ok | 7.21 ms / 7.32 ms | 8.69 ms | 532 MB (rss) | ✓ 0×tol | 1.21x vs nitrix-jax |
| cov | jax-cuda12 | c=500,n_obs=2000 | `cupy.cov` | ok | 3.68 ms / 3.70 ms | 263.00 ms | 4.19 MB (hbm) | ✓ 3e-12×tol | 16.44x vs nitrix-jax |
| cov | jax-cuda12 | c=500,n_obs=2000 | `nitrix-jax` | ok | 224.1 µs / 230.9 µs | 350.64 ms | 83.50 MB (hbm) | ✓ 0.00052×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=500,n_obs=2000 | `numpy.cov` | ok | 7.54 ms / 17.71 ms | 189.62 ms | 4.19 MB (hbm) | ✓ 0×tol | 33.67x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

