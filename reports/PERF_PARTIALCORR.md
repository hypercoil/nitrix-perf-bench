# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: a1af688c287be74a0019e7ceb96677e6ab023820
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T23:15:32.218280+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| partialcorr | jax-cpu | c=128,obs=1024 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=128,obs=1024 | `nilearn.partial_correlation` | ok | 41.75 ms / 162.81 ms | 919.96 ms | 759 MB (rss) | ✓ 9.2e-13×tol | 61.37x vs nitrix-jax |
| partialcorr | jax-cpu | c=128,obs=1024 | `nitrix-jax` | ok | 680.2 µs / 748.7 µs | 272.93 ms | 759 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=128,obs=1024 | `numpy.partialcorr` | ok | 794.9 µs / 804.1 µs | 877.1 µs | 759 MB (rss) | ✓ 0.00022×tol | 1.17x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `cupy.partialcorr` | ok | 733.2 µs / 739.9 µs | 240.04 ms | 0.52 MB (hbm) | ✓ 0.00049×tol | 2.09x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `nilearn.partial_correlation` | ok | 178.52 ms / 250.44 ms | 1.247 s | 0.52 MB (hbm) | ✓ 9.2e-13×tol | 507.83x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `nitrix-jax` | ok | 351.5 µs / 360.6 µs | 577.70 ms | 72.88 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `numpy.partialcorr` | ok | 804.7 µs / 812.1 µs | 896.3 µs | 0.52 MB (hbm) | ✓ 0.00022×tol | 2.29x vs nitrix-jax |
| partialcorr | jax-cpu | c=256,obs=2048 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=256,obs=2048 | `nilearn.partial_correlation` | ok | 59.83 ms / 245.23 ms | 952.55 ms | 759 MB (rss) | ✓ 1.2e-12×tol | 12.57x vs nitrix-jax |
| partialcorr | jax-cpu | c=256,obs=2048 | `nitrix-jax` | ok | 4.76 ms / 8.81 ms | 504.02 ms | 759 MB (rss) | ✓ 0.0013×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=256,obs=2048 | `numpy.partialcorr` | ok | 4.61 ms / 5.15 ms | 4.10 ms | 759 MB (rss) | ✓ 0.00022×tol | 0.97x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `cupy.partialcorr` | ok | 1.38 ms / 1.39 ms | 301.45 ms | 2.10 MB (hbm) | ✓ 0.00066×tol | 2.33x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `nilearn.partial_correlation` | ok | 55.97 ms / 222.97 ms | 937.46 ms | 2.10 MB (hbm) | ✓ 1.2e-12×tol | 94.28x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `nitrix-jax` | ok | 593.6 µs / 604.1 µs | 546.13 ms | 77.59 MB (hbm) | ✓ 0.00058×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `numpy.partialcorr` | ok | 4.15 ms / 497.51 ms | 110.34 ms | 2.10 MB (hbm) | ✓ 0.00022×tol | 6.99x vs nitrix-jax |
| partialcorr | jax-cpu | c=512,obs=4096 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=512,obs=4096 | `nilearn.partial_correlation` | ok | 313.73 ms / 339.97 ms | 1.030 s | 759 MB (rss) | ✓ 1.3e-12×tol | 15.45x vs nitrix-jax |
| partialcorr | jax-cpu | c=512,obs=4096 | `nitrix-jax` | ok | 20.31 ms / 26.11 ms | 2.216 s | 759 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=512,obs=4096 | `numpy.partialcorr` | ok | 26.25 ms / 42.67 ms | 29.55 ms | 759 MB (rss) | ✓ 0.00022×tol | 1.29x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `cupy.partialcorr` | ok | 6.81 ms / 6.85 ms | 282.57 ms | 8.39 MB (hbm) | ✓ 0.00067×tol | 4.25x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `nilearn.partial_correlation` | ok | 315.37 ms / 424.50 ms | 1.390 s | 8.39 MB (hbm) | ✓ 1.3e-12×tol | 196.90x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `nitrix-jax` | ok | 1.60 ms / 1.61 ms | 801.54 ms | 88.08 MB (hbm) | ✓ 0.00074×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `numpy.partialcorr` | ok | 24.38 ms / 30.98 ms | 38.85 ms | 8.39 MB (hbm) | ✓ 0.00022×tol | 15.22x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

