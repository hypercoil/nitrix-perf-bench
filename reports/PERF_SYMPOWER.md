# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: daec36e9ca79a88bd13028edca7a0a02eebbfc7e
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T04:09:07.057054+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sympower | jax-cpu | d=256,power=0.75 | `cupy.eigh_matpow` | skipped | — | — | — | — | — |
| sympower | jax-cpu | d=256,power=0.75 | `nitrix-jax` | ok | 5.28 ms / 23.17 ms | 219.04 ms | 529 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| sympower | jax-cpu | d=256,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 199.00 ms / 314.79 ms | 297.12 ms | 472 MB (rss) | ✓ 0.0017×tol | 37.69x vs nitrix-jax |
| sympower | jax-cuda12 | d=256,power=0.75 | `cupy.eigh_matpow` | skipped | — | — | — | — | — |
| sympower | jax-cuda12 | d=256,power=0.75 | `nitrix-jax` | ok | 1.94 ms / 1.97 ms | 1.240 s | 72.09 MB (hbm) | ✓ 0.00054×tol | 1.00x vs nitrix-jax |
| sympower | jax-cuda12 | d=256,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 147.47 ms / 231.03 ms | 440.22 ms | 0.26 MB (hbm) | ✓ 0.0017×tol | 75.87x vs nitrix-jax |
| sympower | jax-cpu | d=512,power=0.75 | `cupy.eigh_matpow` | skipped | — | — | — | — | — |
| sympower | jax-cpu | d=512,power=0.75 | `nitrix-jax` | ok | 24.27 ms / 29.86 ms | 441.20 ms | 547 MB (rss) | ✓ 0.00067×tol | 1.00x vs nitrix-jax |
| sympower | jax-cpu | d=512,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 656.61 ms / 1.986 s | 1.836 s | 493 MB (rss) | ✓ 0.0017×tol | 27.05x vs nitrix-jax |
| sympower | jax-cuda12 | d=512,power=0.75 | `cupy.eigh_matpow` | skipped | — | — | — | — | — |
| sympower | jax-cuda12 | d=512,power=0.75 | `nitrix-jax` | ok | 4.28 ms / 4.34 ms | 1.238 s | 74.45 MB (hbm) | ✓ 0.00048×tol | 1.00x vs nitrix-jax |
| sympower | jax-cuda12 | d=512,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 493.98 ms / 537.86 ms | 3.474 s | 1.05 MB (hbm) | ✓ 0.0017×tol | 115.38x vs nitrix-jax |
| sympower | jax-cpu | d=64,power=0.75 | `cupy.eigh_matpow` | skipped | — | — | — | — | — |
| sympower | jax-cpu | d=64,power=0.75 | `nitrix-jax` | ok | 392.9 µs / 400.1 µs | 173.70 ms | 518 MB (rss) | ✓ 0.0006×tol | 1.00x vs nitrix-jax |
| sympower | jax-cpu | d=64,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 4.82 ms / 21.33 ms | 181.88 ms | 472 MB (rss) | ✓ 0.0015×tol | 12.25x vs nitrix-jax |
| sympower | jax-cuda12 | d=64,power=0.75 | `cupy.eigh_matpow` | ok | 702.0 µs / 707.7 µs | 553.81 ms | 0.02 MB (hbm) | ✓ 0.00057×tol | 0.94x vs nitrix-jax |
| sympower | jax-cuda12 | d=64,power=0.75 | `nitrix-jax` | ok | 746.3 µs / 758.8 µs | 687.86 ms | 71.35 MB (hbm) | ✓ 0.00057×tol | 1.00x vs nitrix-jax |
| sympower | jax-cuda12 | d=64,power=0.75 | `scipy.linalg.fractional_matrix_power` | ok | 137.60 ms / 187.51 ms | 397.99 ms | 0.02 MB (hbm) | ✓ 0.0015×tol | 184.38x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

