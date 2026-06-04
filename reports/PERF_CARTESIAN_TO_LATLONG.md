# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 143257d8664da2b6629f5397974378019d1b3d68
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T23:27:37.871581+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cartesian_to_latlong | jax-cpu | n=16384 | `cupy.cartesian_to_latlong` | skipped | — | — | — | — | — |
| cartesian_to_latlong | jax-cpu | n=16384 | `nitrix-jax` | ok | 801.7 µs / 826.1 µs | 74.06 ms | 704 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cpu | n=16384 | `numpy.cartesian_to_latlong` | ok | 778.9 µs / 802.4 µs | 825.0 µs | 704 MB (rss) | ✓ 0.00011×tol | 0.97x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=16384 | `cupy.cartesian_to_latlong` | ok | 160.0 µs / 175.9 µs | 148.07 ms | 0.20 MB (hbm) | ✓ 0.00015×tol | 1.74x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=16384 | `nitrix-jax` | ok | 91.9 µs / 102.9 µs | 128.44 ms | 0.46 MB (hbm) | ✓ 0.00016×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=16384 | `numpy.cartesian_to_latlong` | ok | 915.7 µs / 1.04 ms | 1.13 ms | 0.20 MB (hbm) | ✓ 0.00011×tol | 9.96x vs nitrix-jax |
| cartesian_to_latlong | jax-cpu | n=4096 | `cupy.cartesian_to_latlong` | skipped | — | — | — | — | — |
| cartesian_to_latlong | jax-cpu | n=4096 | `nitrix-jax` | ok | 186.9 µs / 195.2 µs | 63.38 ms | 704 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cpu | n=4096 | `numpy.cartesian_to_latlong` | ok | 189.8 µs / 229.6 µs | 288.1 µs | 704 MB (rss) | ✓ 0.00011×tol | 1.02x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=4096 | `cupy.cartesian_to_latlong` | ok | 120.2 µs / 124.6 µs | 1.402 s | 0.05 MB (hbm) | ✓ 0.00013×tol | 1.25x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 95.9 µs / 105.9 µs | 142.46 ms | 0.11 MB (hbm) | ✓ 0.00015×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=4096 | `numpy.cartesian_to_latlong` | ok | 241.2 µs / 246.9 µs | 273.9 µs | 0.05 MB (hbm) | ✓ 0.00011×tol | 2.52x vs nitrix-jax |
| cartesian_to_latlong | jax-cpu | n=65536 | `cupy.cartesian_to_latlong` | skipped | — | — | — | — | — |
| cartesian_to_latlong | jax-cpu | n=65536 | `nitrix-jax` | ok | 1.67 ms / 1.85 ms | 72.61 ms | 704 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cpu | n=65536 | `numpy.cartesian_to_latlong` | ok | 3.12 ms / 3.23 ms | 3.39 ms | 704 MB (rss) | ✓ 0.00012×tol | 1.87x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=65536 | `cupy.cartesian_to_latlong` | ok | 123.2 µs / 130.9 µs | 140.44 ms | 0.79 MB (hbm) | ✓ 0.00016×tol | 1.32x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=65536 | `nitrix-jax` | ok | 93.4 µs / 99.2 µs | 127.80 ms | 2.10 MB (hbm) | ✓ 0.00018×tol | 1.00x vs nitrix-jax |
| cartesian_to_latlong | jax-cuda12 | n=65536 | `numpy.cartesian_to_latlong` | ok | 3.11 ms / 3.14 ms | 3.14 ms | 0.79 MB (hbm) | ✓ 0.00012×tol | 33.32x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

