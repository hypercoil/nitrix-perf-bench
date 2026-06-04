# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 5d1717e0587fe78c6333e685984b5f3315975563 | bench: 361141e23b7656ed9b44d33307996457760afe5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T04:55:04.356638+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| jacobian_det_displacement | jax-cpu | d=32 | `cupy.jacobian_det_displacement` | skipped | — | — | — | — | — |
| jacobian_det_displacement | jax-cpu | d=32 | `nitrix-jax` | ok | 178.4 µs / 181.8 µs | 168.81 ms | 782 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cpu | d=32 | `numpy.jacobian_det` | ok | 890.0 µs / 910.5 µs | 2.43 ms | 782 MB (rss) | ✓ 0.00033×tol | 4.99x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=32 | `cupy.jacobian_det_displacement` | ok | 842.5 µs / 918.2 µs | 164.36 ms | 0.39 MB (hbm) | ✓ 0.00033×tol | 8.06x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=32 | `nitrix-jax` | ok | 104.5 µs / 118.8 µs | 622.01 ms | 38.80 MB (hbm) | ✓ 0.00033×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=32 | `numpy.jacobian_det` | ok | 812.5 µs / 823.2 µs | 2.23 ms | 0.39 MB (hbm) | ✓ 0.00033×tol | 7.77x vs nitrix-jax |
| jacobian_det_displacement | jax-cpu | d=48 | `cupy.jacobian_det_displacement` | skipped | — | — | — | — | — |
| jacobian_det_displacement | jax-cpu | d=48 | `nitrix-jax` | ok | 369.1 µs / 382.9 µs | 196.76 ms | 782 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cpu | d=48 | `numpy.jacobian_det` | ok | 2.48 ms / 2.56 ms | 3.88 ms | 782 MB (rss) | ✓ 0.0003×tol | 6.72x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=48 | `cupy.jacobian_det_displacement` | ok | 826.8 µs / 831.5 µs | 142.08 ms | 2.10 MB (hbm) | ✓ 0.0003×tol | 7.07x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=48 | `nitrix-jax` | ok | 117.0 µs / 127.3 µs | 714.57 ms | 51.58 MB (hbm) | ✓ 0.0003×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=48 | `numpy.jacobian_det` | ok | 2.42 ms / 3.11 ms | 4.20 ms | 2.10 MB (hbm) | ✓ 0.0003×tol | 20.71x vs nitrix-jax |
| jacobian_det_displacement | jax-cpu | d=64 | `cupy.jacobian_det_displacement` | skipped | — | — | — | — | — |
| jacobian_det_displacement | jax-cpu | d=64 | `nitrix-jax` | ok | 854.6 µs / 864.9 µs | 190.74 ms | 782 MB (rss) | ✓ 0.00027×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cpu | d=64 | `numpy.jacobian_det` | ok | 7.30 ms / 7.79 ms | 9.53 ms | 782 MB (rss) | ✓ 0.00033×tol | 8.54x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=64 | `cupy.jacobian_det_displacement` | ok | 828.3 µs / 844.8 µs | 366.65 ms | 4.19 MB (hbm) | ✓ 0.00033×tol | 5.43x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=64 | `nitrix-jax` | ok | 152.5 µs / 154.8 µs | 640.50 ms | 77.59 MB (hbm) | ✓ 0.00033×tol | 1.00x vs nitrix-jax |
| jacobian_det_displacement | jax-cuda12 | d=64 | `numpy.jacobian_det` | ok | 6.81 ms / 6.93 ms | 9.22 ms | 4.19 MB (hbm) | ✓ 0.00033×tol | 44.66x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

