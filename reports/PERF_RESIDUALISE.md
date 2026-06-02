# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 8259a204821af34c01e8092f9cf37275f388413d | bench: 5e5d31ee4e691c0e081dc112fbadc07d6cc03be7
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T23:41:45.423254+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| residualise | jax-cpu | V=1000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=1000,N=400,K=24 | `nilearn.signal_clean` | ok | 20.02 ms / 125.52 ms | 285.61 ms | 2171 MB (rss) | ✓ 0.0045×tol | 22.79x vs nitrix-jax |
| residualise | jax-cpu | V=1000,N=400,K=24 | `nitrix-jax` | ok | 878.3 µs / 1.02 ms | 292.00 ms | 2171 MB (rss) | ✓ 0.0038×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.46 ms / 10.86 ms | 9.54 ms | 2171 MB (rss) | ✓ 0.0017×tol | 8.49x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 1.92 ms / 1.96 ms | 764.43 ms | 2.10 MB (hbm) | ✓ 0.0078×tol | 11.16x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `nilearn.signal_clean` | ok | 23.56 ms / 60.96 ms | 249.19 ms | 2.10 MB (hbm) | ✓ 0.0045×tol | 137.08x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `nitrix-jax` | ok | 171.8 µs / 177.8 µs | 650.65 ms | 76.60 MB (hbm) | ✓ 0.0032×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 7.93 ms / 28.06 ms | 57.20 ms | 2.10 MB (hbm) | ✓ 0.0017×tol | 46.14x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=10000,N=400,K=24 | `nilearn.signal_clean` | ok | 68.46 ms / 138.47 ms | 259.27 ms | 2171 MB (rss) | ✓ 0.0054×tol | 9.52x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `nitrix-jax` | ok | 7.19 ms / 36.95 ms | 609.52 ms | 2171 MB (rss) | ✓ 0.0061×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 90.13 ms / 143.85 ms | 1.156 s | 2171 MB (rss) | ✓ 0.0022×tol | 12.53x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 2.34 ms / 2.38 ms | 278.55 ms | 16.82 MB (hbm) | ✓ 0.0098×tol | 7.04x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `nilearn.signal_clean` | ok | 103.77 ms / 169.64 ms | 326.54 ms | 16.82 MB (hbm) | ✓ 0.0054×tol | 312.03x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `nitrix-jax` | ok | 332.5 µs / 341.2 µs | 978.26 ms | 120.12 MB (hbm) | ✓ 0.0041×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 88.77 ms / 90.10 ms | 91.90 ms | 16.82 MB (hbm) | ✓ 0.0022×tol | 266.95x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `cupy.linalg.lstsq` | skipped | — | — | — | — | — |
| residualise | jax-cpu | V=100000,N=400,K=24 | `nilearn.signal_clean` | ok | 421.74 ms / 434.80 ms | 596.46 ms | 2171 MB (rss) | ✓ 0.0062×tol | 2.01x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `nitrix-jax` | ok | 210.26 ms / 224.71 ms | 457.54 ms | 2171 MB (rss) | ✓ 0.0071×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 1.957 s / 3.791 s | 2.851 s | 2171 MB (rss) | ✓ 0.0025×tol | 9.31x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `cupy.linalg.lstsq` | ok | 13.07 ms / 13.51 ms | 359.12 ms | 268.47 MB (hbm) | ✓ 0.012×tol | 2.34x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `nilearn.signal_clean` | ok | 420.92 ms / 430.81 ms | 622.06 ms | 268.47 MB (hbm) | ✓ 0.0062×tol | 75.39x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `nitrix-jax` | ok | 5.58 ms / 5.66 ms | 960.67 ms | 861.14 MB (hbm) | ✓ 0.0089×tol | 1.00x vs nitrix-jax |
| residualise | jax-cuda12 | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 1.636 s / 3.671 s | 3.105 s | 268.47 MB (hbm) | ✓ 0.0025×tol | 293.05x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

