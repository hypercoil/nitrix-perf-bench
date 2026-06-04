# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 361141e23b7656ed9b44d33307996457760afe5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T04:55:56.945638+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| integrate_velocity_field | jax-cpu | d=16 | `cupy.integrate_velocity_field` | skipped | — | — | — | — | — |
| integrate_velocity_field | jax-cpu | d=16 | `nitrix-jax` | ok | 42.52 ms / 47.00 ms | 2.633 s | 733 MB (rss) | ✓ 0.002×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cpu | d=16 | `scipy.ndimage.map_coordinates` | ok | 4.98 ms / 5.34 ms | 5.10 ms | 704 MB (rss) | ✓ 0.002×tol | 0.12x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=16 | `cupy.integrate_velocity_field` | ok | 1.89 ms / 1.92 ms | 680.93 ms | 0.05 MB (hbm) | ✓ 0.002×tol | 15.96x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=16 | `nitrix-jax` | ok | 118.3 µs / 119.6 µs | 585.93 ms | 0.25 MB (hbm) | ✓ 0.002×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=16 | `scipy.ndimage.map_coordinates` | ok | 9.79 ms / 9.92 ms | 10.19 ms | 0.05 MB (hbm) | ✓ 0.002×tol | 82.77x vs nitrix-jax |
| integrate_velocity_field | jax-cpu | d=24 | `cupy.integrate_velocity_field` | skipped | — | — | — | — | — |
| integrate_velocity_field | jax-cpu | d=24 | `nitrix-jax` | ok | 69.55 ms / 78.08 ms | 3.368 s | 769 MB (rss) | ✓ 0.0041×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cpu | d=24 | `scipy.ndimage.map_coordinates` | ok | 14.52 ms / 16.48 ms | 20.21 ms | 704 MB (rss) | ✓ 0.0041×tol | 0.21x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=24 | `cupy.integrate_velocity_field` | ok | 1.81 ms / 1.85 ms | 173.13 ms | 0.17 MB (hbm) | ✓ 0.0041×tol | 12.84x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=24 | `nitrix-jax` | ok | 141.3 µs / 165.1 µs | 661.69 ms | 0.83 MB (hbm) | ✓ 0.004×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=24 | `scipy.ndimage.map_coordinates` | ok | 14.47 ms / 14.55 ms | 14.78 ms | 0.17 MB (hbm) | ✓ 0.0041×tol | 102.41x vs nitrix-jax |
| integrate_velocity_field | jax-cpu | d=32 | `cupy.integrate_velocity_field` | skipped | — | — | — | — | — |
| integrate_velocity_field | jax-cpu | d=32 | `nitrix-jax` | ok | 351.34 ms / 392.65 ms | 3.090 s | 779 MB (rss) | ✓ 0.0055×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cpu | d=32 | `scipy.ndimage.map_coordinates` | ok | 33.18 ms / 48.00 ms | 39.28 ms | 704 MB (rss) | ✓ 0.0055×tol | 0.09x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=32 | `cupy.integrate_velocity_field` | ok | 1.96 ms / 1.99 ms | 182.83 ms | 0.39 MB (hbm) | ✓ 0.0055×tol | 13.08x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=32 | `nitrix-jax` | ok | 149.6 µs / 153.8 µs | 790.36 ms | 2.10 MB (hbm) | ✓ 0.0055×tol | 1.00x vs nitrix-jax |
| integrate_velocity_field | jax-cuda12 | d=32 | `scipy.ndimage.map_coordinates` | ok | 32.97 ms / 33.24 ms | 33.35 ms | 0.39 MB (hbm) | ✓ 0.0055×tol | 220.43x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

