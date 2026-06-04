# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 3ebfa0efd8d768309c2310eb5850e5e4482e759d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T07:13:26.535189+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| psc_normalize | jax-cpu | n=2048 | `cupy.psc_normalize` | skipped | — | — | — | — | — |
| psc_normalize | jax-cpu | n=2048 | `nitrix-jax` | ok | 1.09 ms / 1.19 ms | 84.35 ms | 1280 MB (rss) | ✓ 0.14×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cpu | n=2048 | `numpy.psc` | ok | 8.37 ms / 14.21 ms | 26.12 ms | 1280 MB (rss) | ✓ 0.15×tol | 7.69x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=2048 | `cupy.psc_normalize` | ok | 287.5 µs / 298.8 µs | 207.22 ms | 16.78 MB (hbm) | ✓ 0.14×tol | 1.87x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 153.7 µs / 164.3 µs | 292.55 ms | 50.35 MB (hbm) | ✓ 0.13×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=2048 | `numpy.psc` | ok | 9.50 ms / 16.34 ms | 26.02 ms | 16.78 MB (hbm) | ✓ 0.15×tol | 61.80x vs nitrix-jax |
| psc_normalize | jax-cpu | n=4096 | `cupy.psc_normalize` | skipped | — | — | — | — | — |
| psc_normalize | jax-cpu | n=4096 | `nitrix-jax` | ok | 20.32 ms / 22.27 ms | 106.75 ms | 1280 MB (rss) | ✓ 0.13×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cpu | n=4096 | `numpy.psc` | ok | 117.47 ms / 121.67 ms | 141.64 ms | 1280 MB (rss) | ✓ 0.14×tol | 5.78x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=4096 | `cupy.psc_normalize` | ok | 1.98 ms / 1.99 ms | 159.56 ms | 67.11 MB (hbm) | ✓ 0.15×tol | 2.12x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 931.1 µs / 959.4 µs | 286.01 ms | 201.36 MB (hbm) | ✓ 0.14×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=4096 | `numpy.psc` | ok | 112.32 ms / 114.92 ms | 108.25 ms | 67.11 MB (hbm) | ✓ 0.14×tol | 120.64x vs nitrix-jax |
| psc_normalize | jax-cpu | n=512 | `cupy.psc_normalize` | skipped | — | — | — | — | — |
| psc_normalize | jax-cpu | n=512 | `nitrix-jax` | ok | 109.7 µs / 115.1 µs | 67.09 ms | 1280 MB (rss) | ✓ 0.095×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cpu | n=512 | `numpy.psc` | ok | 247.9 µs / 282.5 µs | 392.8 µs | 1280 MB (rss) | ✓ 0.13×tol | 2.26x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=512 | `cupy.psc_normalize` | ok | 105.3 µs / 112.7 µs | 161.34 ms | 1.05 MB (hbm) | ✓ 0.13×tol | 1.10x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=512 | `nitrix-jax` | ok | 95.5 µs / 97.0 µs | 256.63 ms | 34.61 MB (hbm) | ✓ 0.095×tol | 1.00x vs nitrix-jax |
| psc_normalize | jax-cuda12 | n=512 | `numpy.psc` | ok | 228.1 µs / 239.7 µs | 295.8 µs | 1.05 MB (hbm) | ✓ 0.13×tol | 2.39x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

