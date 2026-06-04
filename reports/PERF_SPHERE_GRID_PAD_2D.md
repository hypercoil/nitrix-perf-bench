# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: da5314ba815a0132f9e37ccbf3dae4e5832bc8e7
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T06:00:48.010599+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sphere_grid_pad_2d | jax-cpu | h=128 | `cupy.sphere_grid_pad_2d` | skipped | — | — | — | — | — |
| sphere_grid_pad_2d | jax-cpu | h=128 | `nitrix-jax` | ok | 31.7 µs / 34.2 µs | 142.05 ms | 701 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cpu | h=128 | `numpy.sphere_grid_pad` | ok | 30.2 µs / 32.1 µs | 59.9 µs | 701 MB (rss) | ✓ 0×tol | 0.95x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=128 | `cupy.sphere_grid_pad_2d` | ok | 218.6 µs / 228.0 µs | 42.24 ms | 0.13 MB (hbm) | ✓ 0×tol | 2.40x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=128 | `nitrix-jax` | ok | 91.0 µs / 94.0 µs | 130.11 ms | 0.42 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=128 | `numpy.sphere_grid_pad` | ok | 28.9 µs / 29.5 µs | 51.5 µs | 0.13 MB (hbm) | ✓ 0×tol | 0.32x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cpu | h=256 | `cupy.sphere_grid_pad_2d` | skipped | — | — | — | — | — |
| sphere_grid_pad_2d | jax-cpu | h=256 | `nitrix-jax` | ok | 104.3 µs / 129.2 µs | 124.46 ms | 701 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cpu | h=256 | `numpy.sphere_grid_pad` | ok | 62.4 µs / 68.8 µs | 92.3 µs | 701 MB (rss) | ✓ 0×tol | 0.60x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=256 | `cupy.sphere_grid_pad_2d` | ok | 224.0 µs / 231.2 µs | 59.49 ms | 0.52 MB (hbm) | ✓ 0×tol | 2.45x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=256 | `nitrix-jax` | ok | 91.6 µs / 97.0 µs | 154.09 ms | 2.10 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=256 | `numpy.sphere_grid_pad` | ok | 60.9 µs / 62.0 µs | 82.7 µs | 0.52 MB (hbm) | ✓ 0×tol | 0.66x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cpu | h=64 | `cupy.sphere_grid_pad_2d` | skipped | — | — | — | — | — |
| sphere_grid_pad_2d | jax-cpu | h=64 | `nitrix-jax` | ok | 14.9 µs / 15.2 µs | 136.22 ms | 701 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cpu | h=64 | `numpy.sphere_grid_pad` | ok | 34.6 µs / 36.8 µs | 67.2 µs | 701 MB (rss) | ✓ 0×tol | 2.32x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=64 | `cupy.sphere_grid_pad_2d` | ok | 219.9 µs / 234.9 µs | 73.44 ms | 0.03 MB (hbm) | ✓ 0×tol | 1.93x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=64 | `nitrix-jax` | ok | 113.8 µs / 119.8 µs | 140.91 ms | 0.11 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_pad_2d | jax-cuda12 | h=64 | `numpy.sphere_grid_pad` | ok | 20.7 µs / 21.7 µs | 40.7 µs | 0.03 MB (hbm) | ✓ 0×tol | 0.18x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

