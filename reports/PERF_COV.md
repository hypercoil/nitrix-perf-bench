# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T22:44:29.501195+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cov | jax-cpu | c=2000,n_obs=1000 | `nitrix-jax` | ok | 55.53 ms / 70.46 ms | 173.51 ms | 680 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=2000,n_obs=1000 | `numpy.cov` | ok | 66.19 ms / 99.81 ms | 123.16 ms | 575 MB (rss) | ✓ 0×tol | 1.19x vs nitrix-jax |
| cov | jax-cuda12 | c=2000,n_obs=1000 | `nitrix-jax` | ok | 736.3 µs / 748.7 µs | 450.00 ms | 111.69 MB (hbm) | ✓ 0.00081×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=2000,n_obs=1000 | `numpy.cov` | ok | 54.16 ms / 60.88 ms | 93.63 ms | 8.39 MB (hbm) | ✓ 0×tol | 73.56x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `nitrix-jax` | ok | 59.6 µs / 62.1 µs | 135.49 ms | 537 MB (rss) | ✓ 0.00053×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `numpy.cov` | ok | 106.1 µs / 110.7 µs | 45.03 ms | 537 MB (rss) | ✓ 0×tol | 1.78x vs nitrix-jax |
| cov | jax-cuda12 | c=50,n_obs=500 | `nitrix-jax` | ok | 128.7 µs / 159.7 µs | 427.23 ms | 71.56 MB (hbm) | ✓ 0.00015×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=50,n_obs=500 | `numpy.cov` | ok | 108.0 µs / 113.5 µs | 30.95 ms | 0.10 MB (hbm) | ✓ 0×tol | 0.84x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `nitrix-jax` | ok | 11.39 ms / 12.18 ms | 166.20 ms | 540 MB (rss) | ✓ 0.00076×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `numpy.cov` | ok | 7.40 ms / 19.22 ms | 26.35 ms | 537 MB (rss) | ✓ 0×tol | 0.65x vs nitrix-jax |
| cov | jax-cuda12 | c=500,n_obs=2000 | `nitrix-jax` | ok | 216.9 µs / 224.6 µs | 362.91 ms | 83.50 MB (hbm) | ✓ 0.00052×tol | 1.00x vs nitrix-jax |
| cov | jax-cuda12 | c=500,n_obs=2000 | `numpy.cov` | ok | 8.94 ms / 37.23 ms | 58.24 ms | 4.19 MB (hbm) | ✓ 0×tol | 41.22x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

