# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:12:33.151568+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| residualise | jax-cpu | V=1000,N=400,K=24 | `nitrix-jax` | ok | 936.1 µs / 975.2 µs | 475.21 ms | 1817 MB (rss) | ✓ 0.0038×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.79 ms / 14.51 ms | 23.10 ms | 1817 MB (rss) | ✓ 0.0017×tol | 8.32x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `nitrix-jax` | ok | 169.3 µs / 173.0 µs | 729.12 ms | 76.60 MB (hbm) | ✓ 0.0028×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.64 ms / 9.90 ms | 16.22 ms | 2.10 MB (hbm) | ✓ 0.0017×tol | 45.12x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `nitrix-jax` | ok | 10.87 ms / 12.99 ms | 618.44 ms | 1817 MB (rss) | ✓ 0.0061×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 109.46 ms / 162.03 ms | 140.38 ms | 1817 MB (rss) | ✓ 0.0022×tol | 10.07x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `nitrix-jax` | ok | 358.5 µs / 369.2 µs | 975.27 ms | 120.12 MB (hbm) | ✓ 0.0041×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 98.51 ms / 237.55 ms | 1.252 s | 16.82 MB (hbm) | ✓ 0.0022×tol | 274.78x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `nitrix-jax` | ok | 215.36 ms / 223.34 ms | 1.134 s | 1817 MB (rss) | ✓ 0.0071×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 3.042 s / 3.579 s | 2.994 s | 1873 MB (rss) | ✓ 0.0025×tol | 14.12x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `nitrix-jax` | ok | 5.57 ms / 5.61 ms | 663.37 ms | 861.14 MB (hbm) | ✓ 0.0089×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 1.657 s / 3.524 s | 3.929 s | 268.47 MB (hbm) | ✓ 0.0025×tol | 297.68x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

