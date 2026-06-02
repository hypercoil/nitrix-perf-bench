# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: c8c5b5033f3a8aa8dde9a2ea33e5917ad22d3292
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T04:03:21.873307+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| partialcorr | jax-cpu | c=128,obs=1024 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=128,obs=1024 | `nitrix-jax` | ok | 716.1 µs / 827.1 µs | 253.33 ms | 578 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=128,obs=1024 | `numpy.partialcorr` | ok | 810.1 µs / 843.0 µs | 1.31 ms | 568 MB (rss) | ✓ 0.00022×tol | 1.13x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `cupy.partialcorr` | ok | 732.7 µs / 738.1 µs | 928.43 ms | 0.52 MB (hbm) | ✓ 0.00049×tol | 2.19x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `nitrix-jax` | ok | 333.8 µs / 339.4 µs | 614.69 ms | 72.88 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=128,obs=1024 | `numpy.partialcorr` | ok | 824.2 µs / 832.5 µs | 115.40 ms | 0.52 MB (hbm) | ✓ 0.00022×tol | 2.47x vs nitrix-jax |
| partialcorr | jax-cpu | c=256,obs=2048 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=256,obs=2048 | `nitrix-jax` | ok | 22.08 ms / 38.50 ms | 408.97 ms | 596 MB (rss) | ✓ 0.0013×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=256,obs=2048 | `numpy.partialcorr` | ok | 4.75 ms / 5.03 ms | 4.22 ms | 568 MB (rss) | ✓ 0.00022×tol | 0.22x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `cupy.partialcorr` | ok | 1.38 ms / 1.39 ms | 225.22 ms | 2.10 MB (hbm) | ✓ 0.00066×tol | 2.28x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `nitrix-jax` | ok | 605.0 µs / 611.3 µs | 645.62 ms | 77.59 MB (hbm) | ✓ 0.00059×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=256,obs=2048 | `numpy.partialcorr` | ok | 23.11 ms / 877.17 ms | 4.55 ms | 2.10 MB (hbm) | ✓ 0.00022×tol | 38.19x vs nitrix-jax |
| partialcorr | jax-cpu | c=512,obs=4096 | `cupy.partialcorr` | skipped | — | — | — | — | — |
| partialcorr | jax-cpu | c=512,obs=4096 | `nitrix-jax` | ok | 19.27 ms / 40.41 ms | 505.97 ms | 663 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cpu | c=512,obs=4096 | `numpy.partialcorr` | ok | 26.27 ms / 45.38 ms | 32.98 ms | 568 MB (rss) | ✓ 0.00022×tol | 1.36x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `cupy.partialcorr` | ok | 6.81 ms / 6.83 ms | 340.03 ms | 8.39 MB (hbm) | ✓ 0.00067×tol | 4.29x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `nitrix-jax` | ok | 1.59 ms / 1.59 ms | 821.83 ms | 88.08 MB (hbm) | ✓ 0.00074×tol | 1.00x vs nitrix-jax |
| partialcorr | jax-cuda12 | c=512,obs=4096 | `numpy.partialcorr` | ok | 24.16 ms / 27.67 ms | 28.39 ms | 8.39 MB (hbm) | ✓ 0.00022×tol | 15.21x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

