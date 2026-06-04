# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 3ebfa0efd8d768309c2310eb5850e5e4482e759d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T07:12:23.003074+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| zscore_normalize | jax-cpu | n=2048 | `cupy.zscore_normalize` | skipped | — | — | — | — | — |
| zscore_normalize | jax-cpu | n=2048 | `nitrix-jax` | ok | 2.79 ms / 2.89 ms | 147.82 ms | 1280 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cpu | n=2048 | `scipy.stats.zscore` | ok | 14.57 ms / 17.29 ms | 53.83 ms | 1280 MB (rss) | ✓ 0.0011×tol | 5.21x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=2048 | `cupy.zscore_normalize` | ok | 304.0 µs / 308.8 µs | 166.55 ms | 16.78 MB (hbm) | ✓ 0.0011×tol | 2.05x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 148.3 µs / 157.6 µs | 228.03 ms | 50.33 MB (hbm) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=2048 | `scipy.stats.zscore` | ok | 17.49 ms / 18.35 ms | 41.39 ms | 16.78 MB (hbm) | ✓ 0.0011×tol | 117.89x vs nitrix-jax |
| zscore_normalize | jax-cpu | n=4096 | `cupy.zscore_normalize` | skipped | — | — | — | — | — |
| zscore_normalize | jax-cpu | n=4096 | `nitrix-jax` | ok | 27.57 ms / 28.12 ms | 150.56 ms | 1280 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cpu | n=4096 | `scipy.stats.zscore` | ok | 219.03 ms / 254.39 ms | 209.70 ms | 1280 MB (rss) | ✓ 0.001×tol | 7.94x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=4096 | `cupy.zscore_normalize` | ok | 1.94 ms / 1.95 ms | 162.62 ms | 67.11 MB (hbm) | ✓ 0.0012×tol | 2.83x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 684.5 µs / 691.6 µs | 217.56 ms | 201.33 MB (hbm) | ✓ 0.00099×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=4096 | `scipy.stats.zscore` | ok | 206.32 ms / 207.45 ms | 204.10 ms | 67.11 MB (hbm) | ✓ 0.001×tol | 301.40x vs nitrix-jax |
| zscore_normalize | jax-cpu | n=512 | `cupy.zscore_normalize` | skipped | — | — | — | — | — |
| zscore_normalize | jax-cpu | n=512 | `nitrix-jax` | ok | 174.7 µs / 180.2 µs | 133.78 ms | 1280 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cpu | n=512 | `scipy.stats.zscore` | ok | 778.1 µs / 818.4 µs | 1.08 ms | 1280 MB (rss) | ✓ 0.001×tol | 4.45x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=512 | `cupy.zscore_normalize` | ok | 205.5 µs / 214.5 µs | 815.52 ms | 1.05 MB (hbm) | ✓ 0.001×tol | 2.16x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=512 | `nitrix-jax` | ok | 95.1 µs / 109.8 µs | 215.16 ms | 3.15 MB (hbm) | ✓ 0.00082×tol | 1.00x vs nitrix-jax |
| zscore_normalize | jax-cuda12 | n=512 | `scipy.stats.zscore` | ok | 1.15 ms / 1.17 ms | 1.54 ms | 1.05 MB (hbm) | ✓ 0.001×tol | 12.14x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

