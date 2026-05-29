# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:39:15.885931+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| residualise | jax-cpu | V=1000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=1000,N=400,K=24 | `nitrix-jax` | ok | 939.5 µs / 1.07 ms | 395.38 ms | 1818 MB (rss) | ✓ 0.0038×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.63 ms / 13.49 ms | 22.44 ms | 1818 MB (rss) | ✓ 0.0017×tol | 8.13x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 2.00 ms / 2.02 ms | 17.026 s | 2.10 MB (hbm) | ✓ 0.0078×tol | 11.82x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `nitrix-jax` | ok | 169.5 µs / 175.3 µs | 676.15 ms | 76.60 MB (hbm) | ✓ 0.0028×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.90 ms / 21.12 ms | 9.14 ms | 2.10 MB (hbm) | ✓ 0.0017×tol | 46.61x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=10000,N=400,K=24 | `nitrix-jax` | ok | 29.77 ms / 60.09 ms | 717.72 ms | 1818 MB (rss) | ✓ 0.0061×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 100.71 ms / 130.79 ms | 121.60 ms | 1818 MB (rss) | ✓ 0.0022×tol | 3.38x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 2.36 ms / 2.40 ms | 334.78 ms | 16.82 MB (hbm) | ✓ 0.0098×tol | 6.81x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `nitrix-jax` | ok | 347.1 µs / 350.1 µs | 965.91 ms | 120.12 MB (hbm) | ✓ 0.0089×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 136.69 ms / 399.79 ms | 245.84 ms | 16.82 MB (hbm) | ✓ 0.0022×tol | 393.76x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=100000,N=400,K=24 | `nitrix-jax` | ok | 210.13 ms / 220.86 ms | 1.075 s | 1818 MB (rss) | ✓ 0.0071×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 2.865 s / 3.449 s | 2.537 s | 1864 MB (rss) | ✓ 0.0025×tol | 13.64x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 12.81 ms / 13.43 ms | 438.00 ms | 268.47 MB (hbm) | ✓ 0.012×tol | 2.29x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `nitrix-jax` | ok | 5.60 ms / 5.63 ms | 689.56 ms | 861.14 MB (hbm) | ✓ 0.0089×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 1.607 s / 3.359 s | 3.365 s | 268.47 MB (hbm) | ✓ 0.0025×tol | 286.99x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

