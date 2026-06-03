# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 645ce27d898f29997eff5632fb251170ec24d312 | bench: 126d3c00c62903f6e77922b34c777532d2a08b63
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T02:24:43.322063+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gaussian_kernel | jax-cpu | n=2048,d=64 | `cupy.gaussian_kernel` | skipped | — | — | — | — | — |
| gaussian_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 5.90 ms / 10.43 ms | 155.92 ms | 864 MB (rss) | ✓ 0.0069×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cpu | n=2048,d=64 | `sklearn.rbf_kernel` | ok | 32.31 ms / 56.54 ms | 55.49 ms | 864 MB (rss) | ✓ 7.4e-06×tol | 5.47x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=2048,d=64 | `cupy.gaussian_kernel` | ok | 594.3 µs / 602.0 µs | 100.56 ms | 0.52 MB (hbm) | ✓ 6e-05×tol | 3.24x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 183.2 µs / 189.7 µs | 553.37 ms | 105.38 MB (hbm) | ✓ 0.0069×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.rbf_kernel` | ok | 64.85 ms / 97.15 ms | 165.14 ms | 0.52 MB (hbm) | ✓ 7.4e-06×tol | 353.96x vs nitrix-jax |
| gaussian_kernel | jax-cpu | n=4096,d=64 | `cupy.gaussian_kernel` | skipped | — | — | — | — | — |
| gaussian_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 49.30 ms / 55.73 ms | 299.22 ms | 926 MB (rss) | ✓ 0.0069×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cpu | n=4096,d=64 | `sklearn.rbf_kernel` | ok | 196.29 ms / 293.04 ms | 349.69 ms | 864 MB (rss) | ✓ 7.4e-06×tol | 3.98x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=4096,d=64 | `cupy.gaussian_kernel` | ok | 3.86 ms / 3.88 ms | 102.42 ms | 1.05 MB (hbm) | ✓ 6e-05×tol | 4.06x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 951.5 µs / 960.8 µs | 679.14 ms | 206.57 MB (hbm) | ✓ 0.0069×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.rbf_kernel` | ok | 177.67 ms / 190.03 ms | 184.50 ms | 1.05 MB (hbm) | ✓ 7.4e-06×tol | 186.73x vs nitrix-jax |
| gaussian_kernel | jax-cpu | n=512,d=64 | `cupy.gaussian_kernel` | skipped | — | — | — | — | — |
| gaussian_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 362.1 µs / 392.7 µs | 121.55 ms | 864 MB (rss) | ✓ 0.0052×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cpu | n=512,d=64 | `sklearn.rbf_kernel` | ok | 2.19 ms / 2.89 ms | 3.61 ms | 864 MB (rss) | ✓ 2.4e-06×tol | 6.04x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=512,d=64 | `cupy.gaussian_kernel` | ok | 120.7 µs / 126.5 µs | 134.76 ms | 0.13 MB (hbm) | ✓ 3.4e-05×tol | 0.87x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 139.1 µs / 147.3 µs | 350.75 ms | 74.45 MB (hbm) | ✓ 0.0043×tol | 1.00x vs nitrix-jax |
| gaussian_kernel | jax-cuda12 | n=512,d=64 | `sklearn.rbf_kernel` | ok | 2.16 ms / 9.47 ms | 5.32 ms | 0.13 MB (hbm) | ✓ 2.4e-06×tol | 15.53x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

