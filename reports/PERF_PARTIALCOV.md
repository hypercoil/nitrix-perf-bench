# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: c8c5b5033f3a8aa8dde9a2ea33e5917ad22d3292
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T04:02:31.690975+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| partialcov | jax-cpu | c=128,obs=1024 | `cupy.partialcov` | skipped | — | — | — | — | — |
| partialcov | jax-cpu | c=128,obs=1024 | `nitrix-jax` | ok | 579.0 µs / 742.9 µs | 244.60 ms | 576 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cpu | c=128,obs=1024 | `numpy.partialcov` | ok | 781.4 µs / 798.5 µs | 887.7 µs | 567 MB (rss) | ✓ 9.9e-05×tol | 1.35x vs nitrix-jax |
| partialcov | jax-cuda12 | c=128,obs=1024 | `cupy.partialcov` | ok | 723.5 µs / 728.1 µs | 701.10 ms | 0.52 MB (hbm) | ✓ 0.00072×tol | 2.17x vs nitrix-jax |
| partialcov | jax-cuda12 | c=128,obs=1024 | `nitrix-jax` | ok | 334.2 µs / 340.2 µs | 582.10 ms | 72.88 MB (hbm) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cuda12 | c=128,obs=1024 | `numpy.partialcov` | ok | 767.6 µs / 778.9 µs | 7.61 ms | 0.52 MB (hbm) | ✓ 9.9e-05×tol | 2.30x vs nitrix-jax |
| partialcov | jax-cpu | c=256,obs=2048 | `cupy.partialcov` | skipped | — | — | — | — | — |
| partialcov | jax-cpu | c=256,obs=2048 | `nitrix-jax` | ok | 3.99 ms / 13.38 ms | 393.56 ms | 595 MB (rss) | ✓ 0.0017×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cpu | c=256,obs=2048 | `numpy.partialcov` | ok | 3.96 ms / 4.61 ms | 4.03 ms | 567 MB (rss) | ✓ 0.0001×tol | 0.99x vs nitrix-jax |
| partialcov | jax-cuda12 | c=256,obs=2048 | `cupy.partialcov` | ok | 1.37 ms / 1.38 ms | 270.41 ms | 2.10 MB (hbm) | ✓ 0.00077×tol | 2.29x vs nitrix-jax |
| partialcov | jax-cuda12 | c=256,obs=2048 | `nitrix-jax` | ok | 600.0 µs / 603.9 µs | 628.87 ms | 77.59 MB (hbm) | ✓ 0.00091×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cuda12 | c=256,obs=2048 | `numpy.partialcov` | ok | 3.85 ms / 82.14 ms | 4.25 ms | 2.10 MB (hbm) | ✓ 0.0001×tol | 6.42x vs nitrix-jax |
| partialcov | jax-cpu | c=512,obs=4096 | `cupy.partialcov` | skipped | — | — | — | — | — |
| partialcov | jax-cpu | c=512,obs=4096 | `nitrix-jax` | ok | 18.90 ms / 28.51 ms | 2.428 s | 663 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cpu | c=512,obs=4096 | `numpy.partialcov` | ok | 29.64 ms / 43.30 ms | 29.93 ms | 567 MB (rss) | ✓ 0.00011×tol | 1.57x vs nitrix-jax |
| partialcov | jax-cuda12 | c=512,obs=4096 | `cupy.partialcov` | ok | 6.80 ms / 6.82 ms | 270.16 ms | 8.39 MB (hbm) | ✓ 0.0011×tol | 4.29x vs nitrix-jax |
| partialcov | jax-cuda12 | c=512,obs=4096 | `nitrix-jax` | ok | 1.58 ms / 1.62 ms | 792.77 ms | 88.08 MB (hbm) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| partialcov | jax-cuda12 | c=512,obs=4096 | `numpy.partialcov` | ok | 23.84 ms / 24.43 ms | 24.39 ms | 8.39 MB (hbm) | ✓ 0.00011×tol | 15.04x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

