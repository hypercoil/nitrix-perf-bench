# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: da5314ba815a0132f9e37ccbf3dae4e5832bc8e7
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T06:01:32.464946+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sphere_grid_unpad_2d | jax-cpu | h=128 | `cupy.sphere_grid_unpad_2d` | skipped | — | — | — | — | — |
| sphere_grid_unpad_2d | jax-cpu | h=128 | `nitrix-jax` | ok | 13.3 µs / 13.9 µs | 58.89 ms | 699 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cpu | h=128 | `numpy.sphere_grid_unpad` | ok | 0.7 µs / 0.9 µs | 5.1 µs | 699 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=128 | `cupy.sphere_grid_unpad_2d` | ok | 3.4 µs / 3.9 µs | 27.1 µs | 0.13 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=128 | `nitrix-jax` | ok | 89.6 µs / 105.6 µs | 108.64 ms | 0.37 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=128 | `numpy.sphere_grid_unpad` | ok | 0.6 µs / 0.7 µs | 5.0 µs | 0.13 MB (hbm) | ✓ 0×tol | 0.01x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cpu | h=256 | `cupy.sphere_grid_unpad_2d` | skipped | — | — | — | — | — |
| sphere_grid_unpad_2d | jax-cpu | h=256 | `nitrix-jax` | ok | 28.8 µs / 39.9 µs | 61.20 ms | 699 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cpu | h=256 | `numpy.sphere_grid_unpad` | ok | 0.5 µs / 0.6 µs | 4.7 µs | 699 MB (rss) | ✓ 0×tol | 0.02x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=256 | `cupy.sphere_grid_unpad_2d` | ok | 3.4 µs / 3.7 µs | 29.1 µs | 0.52 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=256 | `nitrix-jax` | ok | 95.3 µs / 99.5 µs | 102.76 ms | 1.52 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=256 | `numpy.sphere_grid_unpad` | ok | 0.5 µs / 0.6 µs | 3.8 µs | 0.52 MB (hbm) | ✓ 0×tol | 0.01x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cpu | h=64 | `cupy.sphere_grid_unpad_2d` | skipped | — | — | — | — | — |
| sphere_grid_unpad_2d | jax-cpu | h=64 | `nitrix-jax` | ok | 8.9 µs / 9.3 µs | 49.11 ms | 699 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cpu | h=64 | `numpy.sphere_grid_unpad` | ok | 0.4 µs / 0.5 µs | 4.5 µs | 699 MB (rss) | ✓ 0×tol | 0.05x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=64 | `cupy.sphere_grid_unpad_2d` | ok | 3.3 µs / 3.8 µs | 26.4 µs | 0.03 MB (hbm) | ✓ 0×tol | 0.03x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=64 | `nitrix-jax` | ok | 95.3 µs / 117.8 µs | 99.92 ms | 0.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| sphere_grid_unpad_2d | jax-cuda12 | h=64 | `numpy.sphere_grid_unpad` | ok | 0.4 µs / 0.5 µs | 5.1 µs | 0.03 MB (hbm) | ✓ 0×tol | 0.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

