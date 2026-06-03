# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 2a974f276b865b792fe58c72442757921790fe3c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T22:42:34.316772+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| center_of_mass_points | jax-cpu | p=1024 | `cupy.center_of_mass_points` | skipped | — | — | — | — | — |
| center_of_mass_points | jax-cpu | p=1024 | `nitrix-jax` | ok | 41.8 µs / 43.8 µs | 59.16 ms | 718 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cpu | p=1024 | `numpy.weighted_mean` | ok | 37.2 µs / 38.6 µs | 90.4 µs | 718 MB (rss) | ✓ 0.00018×tol | 0.89x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=1024 | `cupy.center_of_mass_points` | ok | 75.4 µs / 79.5 µs | 145.61 ms | 0.27 MB (hbm) | ✓ 0.00014×tol | 0.68x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=1024 | `nitrix-jax` | ok | 110.3 µs / 113.4 µs | 341.23 ms | 71.58 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=1024 | `numpy.weighted_mean` | ok | 37.3 µs / 37.7 µs | 83.0 µs | 0.27 MB (hbm) | ✓ 0.00018×tol | 0.34x vs nitrix-jax |
| center_of_mass_points | jax-cpu | p=16384 | `cupy.center_of_mass_points` | skipped | — | — | — | — | — |
| center_of_mass_points | jax-cpu | p=16384 | `nitrix-jax` | ok | 367.0 µs / 460.6 µs | 139.82 ms | 718 MB (rss) | ✓ 0.00031×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cpu | p=16384 | `numpy.weighted_mean` | ok | 477.7 µs / 487.9 µs | 736.3 µs | 718 MB (rss) | ✓ 8.3e-05×tol | 1.30x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=16384 | `cupy.center_of_mass_points` | ok | 76.0 µs / 80.3 µs | 138.45 ms | 4.39 MB (hbm) | ✓ 6e-05×tol | 0.57x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=16384 | `nitrix-jax` | ok | 134.1 µs / 157.5 µs | 433.35 ms | 75.74 MB (hbm) | ✓ 4.7e-05×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=16384 | `numpy.weighted_mean` | ok | 479.5 µs / 483.7 µs | 603.5 µs | 4.39 MB (hbm) | ✓ 8.3e-05×tol | 3.58x vs nitrix-jax |
| center_of_mass_points | jax-cpu | p=4096 | `cupy.center_of_mass_points` | skipped | — | — | — | — | — |
| center_of_mass_points | jax-cpu | p=4096 | `nitrix-jax` | ok | 105.3 µs / 122.0 µs | 104.06 ms | 718 MB (rss) | ✓ 0.00034×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cpu | p=4096 | `numpy.weighted_mean` | ok | 163.7 µs / 171.1 µs | 253.9 µs | 718 MB (rss) | ✓ 0.00015×tol | 1.55x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=4096 | `cupy.center_of_mass_points` | ok | 75.9 µs / 80.6 µs | 140.78 ms | 1.10 MB (hbm) | ✓ 8.3e-05×tol | 0.70x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=4096 | `nitrix-jax` | ok | 108.8 µs / 112.3 µs | 478.74 ms | 72.43 MB (hbm) | ✓ 7.8e-05×tol | 1.00x vs nitrix-jax |
| center_of_mass_points | jax-cuda12 | p=4096 | `numpy.weighted_mean` | ok | 170.2 µs / 171.1 µs | 254.8 µs | 1.10 MB (hbm) | ✓ 0.00015×tol | 1.56x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

