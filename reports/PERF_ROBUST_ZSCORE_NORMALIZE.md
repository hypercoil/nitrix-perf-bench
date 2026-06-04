# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 3ebfa0efd8d768309c2310eb5850e5e4482e759d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T07:14:22.925879+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| robust_zscore_normalize | jax-cpu | n=2048 | `cupy.robust_zscore_normalize` | skipped | — | — | — | — | — |
| robust_zscore_normalize | jax-cpu | n=2048 | `nitrix-jax` | ok | 1.178 s / 1.183 s | 1.448 s | 1280 MB (rss) | ✓ 0.00043×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cpu | n=2048 | `numpy.robust_zscore` | ok | 102.51 ms / 103.90 ms | 139.87 ms | 1280 MB (rss) | ✓ 0.00043×tol | 0.09x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=2048 | `cupy.robust_zscore_normalize` | ok | 17.98 ms / 18.01 ms | 4.951 s | 16.78 MB (hbm) | ✓ 0.00043×tol | 12.84x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 1.40 ms / 1.42 ms | 1.280 s | 83.89 MB (hbm) | ✓ 0.00043×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=2048 | `numpy.robust_zscore` | ok | 101.89 ms / 103.54 ms | 139.33 ms | 16.78 MB (hbm) | ✓ 0.00043×tol | 72.77x vs nitrix-jax |
| robust_zscore_normalize | jax-cpu | n=4096 | `cupy.robust_zscore_normalize` | skipped | — | — | — | — | — |
| robust_zscore_normalize | jax-cpu | n=4096 | `nitrix-jax` | ok | 5.245 s / 5.252 s | 5.711 s | 1280 MB (rss) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cpu | n=4096 | `numpy.robust_zscore` | ok | 603.49 ms / 620.27 ms | 606.46 ms | 1280 MB (rss) | ✓ 0.00042×tol | 0.12x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=4096 | `cupy.robust_zscore_normalize` | ok | 73.93 ms / 74.01 ms | 312.10 ms | 67.11 MB (hbm) | ✓ 0.00042×tol | 9.16x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 8.07 ms / 8.66 ms | 1.360 s | 335.54 MB (hbm) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=4096 | `numpy.robust_zscore` | ok | 594.26 ms / 598.39 ms | 603.80 ms | 67.11 MB (hbm) | ✓ 0.00042×tol | 73.60x vs nitrix-jax |
| robust_zscore_normalize | jax-cpu | n=512 | `cupy.robust_zscore_normalize` | skipped | — | — | — | — | — |
| robust_zscore_normalize | jax-cpu | n=512 | `nitrix-jax` | ok | 60.88 ms / 61.65 ms | 223.51 ms | 1280 MB (rss) | ✓ 0.00045×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cpu | n=512 | `numpy.robust_zscore` | ok | 6.45 ms / 6.57 ms | 6.76 ms | 1280 MB (rss) | ✓ 0.00045×tol | 0.11x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=512 | `cupy.robust_zscore_normalize` | ok | 511.8 µs / 513.3 µs | 6.259 s | 1.05 MB (hbm) | ✓ 0.00045×tol | 2.84x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=512 | `nitrix-jax` | ok | 180.2 µs / 190.0 µs | 986.94 ms | 68.16 MB (hbm) | ✓ 0.00045×tol | 1.00x vs nitrix-jax |
| robust_zscore_normalize | jax-cuda12 | n=512 | `numpy.robust_zscore` | ok | 6.83 ms / 7.03 ms | 6.95 ms | 1.05 MB (hbm) | ✓ 0.00045×tol | 37.90x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

