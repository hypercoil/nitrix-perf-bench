# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:12:07.731970+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| corr | jax-cpu | n=2000,t=1000 | `nitrix-jax` | ok | 43.18 ms / 49.12 ms | 254.91 ms | 700 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=2000,t=1000 | `numpy.corrcoef` | ok | 78.31 ms / 101.25 ms | 89.26 ms | 575 MB (rss) | ✓ 0×tol | 1.81x vs nitrix-jax |
| corr | jax-cuda12 | n=2000,t=1000 | `nitrix-jax` | ok | 752.4 µs / 773.5 µs | 607.18 ms | 111.69 MB (hbm) | ✓ 0.00082×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=2000,t=1000 | `numpy.corrcoef` | ok | 60.82 ms / 61.70 ms | 83.51 ms | 8.39 MB (hbm) | ✓ 0×tol | 80.83x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `nitrix-jax` | ok | 61.2 µs / 62.7 µs | 145.89 ms | 531 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `numpy.corrcoef` | ok | 126.2 µs / 129.1 µs | 210.9 µs | 531 MB (rss) | ✓ 0×tol | 2.06x vs nitrix-jax |
| corr | jax-cuda12 | n=50,t=500 | `nitrix-jax` | ok | 109.8 µs / 111.7 µs | 387.75 ms | 71.56 MB (hbm) | ✓ 0.00022×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=50,t=500 | `numpy.corrcoef` | ok | 126.1 µs / 127.2 µs | 257.2 µs | 0.10 MB (hbm) | ✓ 0×tol | 1.15x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `nitrix-jax` | ok | 6.88 ms / 9.54 ms | 165.56 ms | 562 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `numpy.corrcoef` | ok | 8.06 ms / 17.43 ms | 10.14 ms | 531 MB (rss) | ✓ 0×tol | 1.17x vs nitrix-jax |
| corr | jax-cuda12 | n=500,t=2000 | `nitrix-jax` | ok | 226.5 µs / 233.2 µs | 474.80 ms | 83.50 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| corr | jax-cuda12 | n=500,t=2000 | `numpy.corrcoef` | ok | 7.81 ms / 25.65 ms | 8.00 ms | 4.19 MB (hbm) | ✓ 0×tol | 34.47x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

