# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 361141e23b7656ed9b44d33307996457760afe5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T04:51:56.917865+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| jacobian_displacement | jax-cpu | d=32 | `cupy.jacobian_displacement` | skipped | — | — | — | — | — |
| jacobian_displacement | jax-cpu | d=32 | `nitrix-jax` | ok | 125.4 µs / 134.8 µs | 409.67 ms | 787 MB (rss) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cpu | d=32 | `numpy.jacobian` | ok | 598.0 µs / 665.2 µs | 886.9 µs | 787 MB (rss) | ✓ 5.5e-05×tol | 4.77x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=32 | `cupy.jacobian_displacement` | ok | 714.5 µs / 732.3 µs | 5.681 s | 0.39 MB (hbm) | ✓ 5.5e-05×tol | 7.82x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=32 | `nitrix-jax` | ok | 91.4 µs / 94.8 µs | 525.23 ms | 3.28 MB (hbm) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=32 | `numpy.jacobian` | ok | 595.1 µs / 608.0 µs | 2.25 ms | 0.39 MB (hbm) | ✓ 5.5e-05×tol | 6.51x vs nitrix-jax |
| jacobian_displacement | jax-cpu | d=48 | `cupy.jacobian_displacement` | skipped | — | — | — | — | — |
| jacobian_displacement | jax-cpu | d=48 | `nitrix-jax` | ok | 275.2 µs / 277.9 µs | 304.99 ms | 787 MB (rss) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cpu | d=48 | `numpy.jacobian` | ok | 2.05 ms / 2.32 ms | 8.75 ms | 787 MB (rss) | ✓ 5.5e-05×tol | 7.47x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=48 | `cupy.jacobian_displacement` | ok | 678.0 µs / 715.2 µs | 1.923 s | 2.10 MB (hbm) | ✓ 5.5e-05×tol | 7.05x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=48 | `nitrix-jax` | ok | 96.2 µs / 104.1 µs | 563.25 ms | 10.27 MB (hbm) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=48 | `numpy.jacobian` | ok | 1.83 ms / 1.90 ms | 7.13 ms | 2.10 MB (hbm) | ✓ 5.5e-05×tol | 19.06x vs nitrix-jax |
| jacobian_displacement | jax-cpu | d=64 | `cupy.jacobian_displacement` | skipped | — | — | — | — | — |
| jacobian_displacement | jax-cpu | d=64 | `nitrix-jax` | ok | 756.2 µs / 795.8 µs | 175.74 ms | 787 MB (rss) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cpu | d=64 | `numpy.jacobian` | ok | 5.15 ms / 5.48 ms | 16.60 ms | 787 MB (rss) | ✓ 5.5e-05×tol | 6.81x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=64 | `cupy.jacobian_displacement` | ok | 660.4 µs / 685.8 µs | 145.96 ms | 4.19 MB (hbm) | ✓ 5.5e-05×tol | 5.56x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=64 | `nitrix-jax` | ok | 118.8 µs / 138.7 µs | 170.45 ms | 37.75 MB (hbm) | ✓ 5.5e-05×tol | 1.00x vs nitrix-jax |
| jacobian_displacement | jax-cuda12 | d=64 | `numpy.jacobian` | ok | 4.70 ms / 5.03 ms | 15.47 ms | 4.19 MB (hbm) | ✓ 5.5e-05×tol | 39.58x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

