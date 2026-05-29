# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: daec36e9ca79a88bd13028edca7a0a02eebbfc7e
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T04:08:02.149903+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| symsqrt | jax-cpu | d=256 | `cupy.eigh_sqrtm` | skipped | — | — | — | — | — |
| symsqrt | jax-cpu | d=256 | `nitrix-jax` | ok | 5.11 ms / 6.29 ms | 199.96 ms | 527 MB (rss) | ✓ 0.00044×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cpu | d=256 | `scipy.linalg.sqrtm` | ok | 19.98 ms / 47.16 ms | 131.27 ms | 471 MB (rss) | ✓ 0.0011×tol | 3.91x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=256 | `cupy.eigh_sqrtm` | skipped | — | — | — | — | — |
| symsqrt | jax-cuda12 | d=256 | `nitrix-jax` | ok | 1.97 ms / 1.99 ms | 820.95 ms | 72.09 MB (hbm) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=256 | `scipy.linalg.sqrtm` | ok | 44.54 ms / 180.83 ms | 3.892 s | 0.26 MB (hbm) | ✓ 0.0011×tol | 22.63x vs nitrix-jax |
| symsqrt | jax-cpu | d=512 | `cupy.eigh_sqrtm` | skipped | — | — | — | — | — |
| symsqrt | jax-cpu | d=512 | `nitrix-jax` | ok | 21.82 ms / 167.38 ms | 948.52 ms | 545 MB (rss) | ✓ 0.00068×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cpu | d=512 | `scipy.linalg.sqrtm` | ok | 169.20 ms / 216.27 ms | 292.35 ms | 473 MB (rss) | ✓ 0.0014×tol | 7.75x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=512 | `cupy.eigh_sqrtm` | skipped | — | — | — | — | — |
| symsqrt | jax-cuda12 | d=512 | `nitrix-jax` | ok | 4.30 ms / 4.32 ms | 1.072 s | 74.45 MB (hbm) | ✓ 0.00049×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=512 | `scipy.linalg.sqrtm` | ok | 120.89 ms / 121.39 ms | 221.63 ms | 1.05 MB (hbm) | ✓ 0.0014×tol | 28.10x vs nitrix-jax |
| symsqrt | jax-cpu | d=64 | `cupy.eigh_sqrtm` | skipped | — | — | — | — | — |
| symsqrt | jax-cpu | d=64 | `nitrix-jax` | ok | 322.1 µs / 328.7 µs | 173.23 ms | 517 MB (rss) | ✓ 0.00044×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cpu | d=64 | `scipy.linalg.sqrtm` | ok | 1.06 ms / 1.09 ms | 1.25 ms | 471 MB (rss) | ✓ 0.0011×tol | 3.30x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=64 | `cupy.eigh_sqrtm` | ok | 704.3 µs / 706.2 µs | 553.56 ms | 0.02 MB (hbm) | ✓ 0.00049×tol | 0.94x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=64 | `nitrix-jax` | ok | 751.6 µs / 755.6 µs | 526.15 ms | 71.35 MB (hbm) | ✓ 0.00047×tol | 1.00x vs nitrix-jax |
| symsqrt | jax-cuda12 | d=64 | `scipy.linalg.sqrtm` | ok | 1.07 ms / 1.08 ms | 1.24 ms | 0.02 MB (hbm) | ✓ 0.0011×tol | 1.42x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

