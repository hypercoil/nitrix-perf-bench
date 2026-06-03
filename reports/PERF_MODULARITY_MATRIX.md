# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 88a23053af93d9466c3993dae0309eddd5c11c6f | bench: 698e0cc7813185a23f54be34e95717648b37022e
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T04:40:11.645550+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| modularity_matrix | jax-cpu | n=128 | `cupy.modularity_matrix` | skipped | — | — | — | — | — |
| modularity_matrix | jax-cpu | n=128 | `networkx.modularity_matrix` | ok | 3.09 ms / 3.24 ms | 654.23 ms | 707 MB (rss) | ✓ 0×tol | 88.28x vs nitrix-jax |
| modularity_matrix | jax-cpu | n=128 | `nitrix-jax` | ok | 35.0 µs / 37.1 µs | 115.16 ms | 707 MB (rss) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=128 | `cupy.modularity_matrix` | ok | 163.1 µs / 191.6 µs | 835.14 ms | 0.07 MB (hbm) | ✓ 0.00013×tol | 1.54x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=128 | `networkx.modularity_matrix` | ok | 2.96 ms / 3.20 ms | 386.13 ms | 0.07 MB (hbm) | ✓ 0×tol | 27.93x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=128 | `nitrix-jax` | ok | 106.1 µs / 110.7 µs | 356.35 ms | 33.75 MB (hbm) | ✓ 0.00016×tol | 1.00x vs nitrix-jax |
| modularity_matrix | jax-cpu | n=256 | `cupy.modularity_matrix` | skipped | — | — | — | — | — |
| modularity_matrix | jax-cpu | n=256 | `networkx.modularity_matrix` | ok | 12.60 ms / 18.38 ms | 445.99 ms | 707 MB (rss) | ✓ 0×tol | 142.72x vs nitrix-jax |
| modularity_matrix | jax-cpu | n=256 | `nitrix-jax` | ok | 88.3 µs / 140.6 µs | 86.51 ms | 707 MB (rss) | ✓ 0.00026×tol | 1.00x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=256 | `cupy.modularity_matrix` | ok | 127.6 µs / 133.8 µs | 137.13 ms | 0.26 MB (hbm) | ✓ 0.0002×tol | 1.18x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=256 | `networkx.modularity_matrix` | ok | 12.97 ms / 15.77 ms | 412.99 ms | 0.26 MB (hbm) | ✓ 0×tol | 119.54x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=256 | `nitrix-jax` | ok | 108.5 µs / 124.9 µs | 390.58 ms | 33.82 MB (hbm) | ✓ 0.00015×tol | 1.00x vs nitrix-jax |
| modularity_matrix | jax-cpu | n=512 | `cupy.modularity_matrix` | skipped | — | — | — | — | — |
| modularity_matrix | jax-cpu | n=512 | `networkx.modularity_matrix` | ok | 52.14 ms / 63.58 ms | 480.82 ms | 707 MB (rss) | ✓ 0×tol | 340.95x vs nitrix-jax |
| modularity_matrix | jax-cpu | n=512 | `nitrix-jax` | ok | 152.9 µs / 205.5 µs | 106.73 ms | 707 MB (rss) | ✓ 0.00026×tol | 1.00x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=512 | `cupy.modularity_matrix` | ok | 125.0 µs / 131.2 µs | 152.10 ms | 1.05 MB (hbm) | ✓ 0.00023×tol | 1.18x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=512 | `networkx.modularity_matrix` | ok | 50.81 ms / 53.73 ms | 486.99 ms | 1.05 MB (hbm) | ✓ 0×tol | 481.47x vs nitrix-jax |
| modularity_matrix | jax-cuda12 | n=512 | `nitrix-jax` | ok | 105.5 µs / 114.9 µs | 381.59 ms | 34.61 MB (hbm) | ✓ 0.00019×tol | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

