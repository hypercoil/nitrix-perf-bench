# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 2a974f276b865b792fe58c72442757921790fe3c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T22:41:37.884391+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| center_of_mass_grid | jax-cpu | d=32 | `cupy.center_of_mass_grid` | skipped | — | — | — | — | — |
| center_of_mass_grid | jax-cpu | d=32 | `nitrix-jax` | ok | 53.7 µs / 55.5 µs | 139.15 ms | 722 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cpu | d=32 | `scipy.ndimage.center_of_mass` | ok | 125.5 µs / 134.6 µs | 199.2 µs | 722 MB (rss) | ✓ 6.7e-06×tol | 2.34x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=32 | `cupy.center_of_mass_grid` | ok | 286.0 µs / 293.9 µs | 1.038 s | 0.13 MB (hbm) | ✓ 3.8e-05×tol | 2.59x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=32 | `nitrix-jax` | ok | 110.3 µs / 113.3 µs | 317.58 ms | 33.69 MB (hbm) | ✓ 0.0001×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=32 | `scipy.ndimage.center_of_mass` | ok | 193.1 µs / 229.3 µs | 309.8 µs | 0.13 MB (hbm) | ✓ 6.7e-06×tol | 1.75x vs nitrix-jax |
| center_of_mass_grid | jax-cpu | d=64 | `cupy.center_of_mass_grid` | skipped | — | — | — | — | — |
| center_of_mass_grid | jax-cpu | d=64 | `nitrix-jax` | ok | 249.2 µs / 316.1 µs | 188.33 ms | 722 MB (rss) | ✓ 0.00033×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cpu | d=64 | `scipy.ndimage.center_of_mass` | ok | 752.1 µs / 791.2 µs | 1.59 ms | 722 MB (rss) | ✓ 2.6e-06×tol | 3.02x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=64 | `cupy.center_of_mass_grid` | ok | 287.6 µs / 307.2 µs | 81.08 ms | 1.05 MB (hbm) | ✓ 3.7e-05×tol | 2.63x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=64 | `nitrix-jax` | ok | 109.2 µs / 112.3 µs | 363.93 ms | 34.60 MB (hbm) | ✓ 3.7e-05×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=64 | `scipy.ndimage.center_of_mass` | ok | 989.5 µs / 1.03 ms | 2.04 ms | 1.05 MB (hbm) | ✓ 2.6e-06×tol | 9.06x vs nitrix-jax |
| center_of_mass_grid | jax-cpu | d=96 | `cupy.center_of_mass_grid` | skipped | — | — | — | — | — |
| center_of_mass_grid | jax-cpu | d=96 | `nitrix-jax` | ok | 737.1 µs / 1.23 ms | 208.98 ms | 722 MB (rss) | ✓ 0.00078×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cpu | d=96 | `scipy.ndimage.center_of_mass` | ok | 2.04 ms / 2.14 ms | 4.75 ms | 722 MB (rss) | ✓ 8.6e-06×tol | 2.76x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=96 | `cupy.center_of_mass_grid` | ok | 293.7 µs / 307.8 µs | 69.12 ms | 4.19 MB (hbm) | ✓ 8.6e-05×tol | 2.35x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=96 | `nitrix-jax` | ok | 125.2 µs / 138.2 µs | 383.74 ms | 37.75 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| center_of_mass_grid | jax-cuda12 | d=96 | `scipy.ndimage.center_of_mass` | ok | 2.02 ms / 2.05 ms | 4.84 ms | 4.19 MB (hbm) | ✓ 8.6e-06×tol | 16.11x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

