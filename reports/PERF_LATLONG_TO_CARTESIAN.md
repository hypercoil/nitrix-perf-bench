# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 143257d8664da2b6629f5397974378019d1b3d68
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T23:26:50.237293+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| latlong_to_cartesian | jax-cpu | n=16384 | `cupy.latlong_to_cartesian` | skipped | — | — | — | — | — |
| latlong_to_cartesian | jax-cpu | n=16384 | `nitrix-jax` | ok | 599.7 µs / 659.9 µs | 88.00 ms | 703 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cpu | n=16384 | `numpy.latlong_to_cartesian` | ok | 198.8 µs / 212.5 µs | 222.3 µs | 703 MB (rss) | ✓ 0.00014×tol | 0.33x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=16384 | `cupy.latlong_to_cartesian` | ok | 140.0 µs / 148.5 µs | 47.31 ms | 0.13 MB (hbm) | ✓ 0.00016×tol | 1.46x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=16384 | `nitrix-jax` | ok | 95.9 µs / 102.1 µs | 172.42 ms | 0.52 MB (hbm) | ✓ 0.00016×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=16384 | `numpy.latlong_to_cartesian` | ok | 194.3 µs / 197.1 µs | 226.3 µs | 0.13 MB (hbm) | ✓ 0.00014×tol | 2.03x vs nitrix-jax |
| latlong_to_cartesian | jax-cpu | n=4096 | `cupy.latlong_to_cartesian` | skipped | — | — | — | — | — |
| latlong_to_cartesian | jax-cpu | n=4096 | `nitrix-jax` | ok | 72.2 µs / 74.7 µs | 72.40 ms | 703 MB (rss) | ✓ 0.0001×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cpu | n=4096 | `numpy.latlong_to_cartesian` | ok | 94.4 µs / 96.6 µs | 80.5 µs | 703 MB (rss) | ✓ 0.00013×tol | 1.31x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=4096 | `cupy.latlong_to_cartesian` | ok | 196.7 µs / 214.3 µs | 672.32 ms | 0.03 MB (hbm) | ✓ 0.00014×tol | 2.17x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 90.7 µs / 97.5 µs | 183.38 ms | 0.13 MB (hbm) | ✓ 0.00014×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=4096 | `numpy.latlong_to_cartesian` | ok | 55.3 µs / 57.8 µs | 79.5 µs | 0.03 MB (hbm) | ✓ 0.00013×tol | 0.61x vs nitrix-jax |
| latlong_to_cartesian | jax-cpu | n=65536 | `cupy.latlong_to_cartesian` | skipped | — | — | — | — | — |
| latlong_to_cartesian | jax-cpu | n=65536 | `nitrix-jax` | ok | 1.33 ms / 1.55 ms | 77.17 ms | 703 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cpu | n=65536 | `numpy.latlong_to_cartesian` | ok | 806.9 µs / 827.6 µs | 895.2 µs | 703 MB (rss) | ✓ 0.00015×tol | 0.61x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=65536 | `cupy.latlong_to_cartesian` | ok | 138.8 µs / 149.9 µs | 51.46 ms | 0.52 MB (hbm) | ✓ 0.00017×tol | 1.47x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=65536 | `nitrix-jax` | ok | 94.7 µs / 124.2 µs | 162.71 ms | 2.36 MB (hbm) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| latlong_to_cartesian | jax-cuda12 | n=65536 | `numpy.latlong_to_cartesian` | ok | 740.4 µs / 751.5 µs | 770.8 µs | 0.52 MB (hbm) | ✓ 0.00015×tol | 7.82x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

