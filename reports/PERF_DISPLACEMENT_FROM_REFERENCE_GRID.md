# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 2a974f276b865b792fe58c72442757921790fe3c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T22:43:22.099831+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| displacement_from_reference_grid | jax-cpu | d=32 | `cupy.displacement_from_reference_grid` | skipped | — | — | — | — | — |
| displacement_from_reference_grid | jax-cpu | d=32 | `nitrix-jax` | ok | 48.9 µs / 62.1 µs | 131.25 ms | 721 MB (rss) | ✓ 0.15×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cpu | d=32 | `scipy.ndimage.center_of_mass` | ok | 139.2 µs / 186.7 µs | 281.4 µs | 721 MB (rss) | ✓ 0.00094×tol | 2.85x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=32 | `cupy.displacement_from_reference_grid` | ok | 372.7 µs / 387.2 µs | 361.66 ms | 0.13 MB (hbm) | ✓ 0.0049×tol | 3.27x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=32 | `nitrix-jax` | ok | 114.1 µs / 118.4 µs | 338.72 ms | 33.69 MB (hbm) | ✓ 0.0089×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=32 | `scipy.ndimage.center_of_mass` | ok | 209.4 µs / 220.7 µs | 283.2 µs | 0.13 MB (hbm) | ✓ 0.00094×tol | 1.84x vs nitrix-jax |
| displacement_from_reference_grid | jax-cpu | d=64 | `cupy.displacement_from_reference_grid` | skipped | — | — | — | — | — |
| displacement_from_reference_grid | jax-cpu | d=64 | `nitrix-jax` | ok | 357.3 µs / 398.7 µs | 177.10 ms | 721 MB (rss) | ✓ 0.1×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cpu | d=64 | `scipy.ndimage.center_of_mass` | ok | 809.9 µs / 860.8 µs | 1.77 ms | 721 MB (rss) | ✓ 0.00079×tol | 2.27x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=64 | `cupy.displacement_from_reference_grid` | ok | 361.3 µs / 373.5 µs | 72.34 ms | 1.05 MB (hbm) | ✓ 0.011×tol | 3.39x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=64 | `nitrix-jax` | ok | 106.7 µs / 110.2 µs | 360.47 ms | 34.60 MB (hbm) | ✓ 0.011×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=64 | `scipy.ndimage.center_of_mass` | ok | 738.8 µs / 1.04 ms | 2.09 ms | 1.05 MB (hbm) | ✓ 0.00079×tol | 6.93x vs nitrix-jax |
| displacement_from_reference_grid | jax-cpu | d=96 | `cupy.displacement_from_reference_grid` | skipped | — | — | — | — | — |
| displacement_from_reference_grid | jax-cpu | d=96 | `nitrix-jax` | ok | 675.8 µs / 875.1 µs | 183.64 ms | 721 MB (rss) | ✓ 0.33×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cpu | d=96 | `scipy.ndimage.center_of_mass` | ok | 2.21 ms / 2.38 ms | 5.63 ms | 721 MB (rss) | ✓ 0.0041×tol | 3.28x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=96 | `cupy.displacement_from_reference_grid` | ok | 369.6 µs / 383.3 µs | 71.26 ms | 4.19 MB (hbm) | ✓ 0.041×tol | 3.29x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=96 | `nitrix-jax` | ok | 112.4 µs / 114.7 µs | 387.36 ms | 37.75 MB (hbm) | ✓ 0.045×tol | 1.00x vs nitrix-jax |
| displacement_from_reference_grid | jax-cuda12 | d=96 | `scipy.ndimage.center_of_mass` | ok | 2.11 ms / 2.16 ms | 5.89 ms | 4.19 MB (hbm) | ✓ 0.0041×tol | 18.80x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

