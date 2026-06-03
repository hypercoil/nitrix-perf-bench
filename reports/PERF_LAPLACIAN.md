# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 88a23053af93d9466c3993dae0309eddd5c11c6f | bench: 698e0cc7813185a23f54be34e95717648b37022e
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T04:39:28.516856+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| laplacian | jax-cpu | n=128 | `cupy.laplacian` | skipped | — | — | — | — | — |
| laplacian | jax-cpu | n=128 | `nitrix-jax` | ok | 17.5 µs / 33.2 µs | 119.96 ms | 704 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cpu | n=128 | `scipy.csgraph.laplacian` | ok | 14.0 µs / 15.2 µs | 45.0 µs | 704 MB (rss) | ✓ 0.00022×tol | 0.80x vs nitrix-jax |
| laplacian | jax-cuda12 | n=128 | `cupy.laplacian` | ok | 75.7 µs / 78.8 µs | 389.49 ms | 0.07 MB (hbm) | ✓ 0.00012×tol | 0.76x vs nitrix-jax |
| laplacian | jax-cuda12 | n=128 | `nitrix-jax` | ok | 99.0 µs / 100.6 µs | 155.33 ms | 0.20 MB (hbm) | ✓ 0.00022×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cuda12 | n=128 | `scipy.csgraph.laplacian` | ok | 19.5 µs / 20.7 µs | 58.2 µs | 0.07 MB (hbm) | ✓ 0.00022×tol | 0.20x vs nitrix-jax |
| laplacian | jax-cpu | n=256 | `cupy.laplacian` | skipped | — | — | — | — | — |
| laplacian | jax-cpu | n=256 | `nitrix-jax` | ok | 64.9 µs / 70.7 µs | 100.41 ms | 704 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cpu | n=256 | `scipy.csgraph.laplacian` | ok | 52.6 µs / 55.3 µs | 81.3 µs | 704 MB (rss) | ✓ 0.00026×tol | 0.81x vs nitrix-jax |
| laplacian | jax-cuda12 | n=256 | `cupy.laplacian` | ok | 73.8 µs / 78.4 µs | 61.00 ms | 0.26 MB (hbm) | ✓ 0.00013×tol | 0.74x vs nitrix-jax |
| laplacian | jax-cuda12 | n=256 | `nitrix-jax` | ok | 99.3 µs / 100.7 µs | 221.39 ms | 33.82 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cuda12 | n=256 | `scipy.csgraph.laplacian` | ok | 35.5 µs / 37.2 µs | 64.9 µs | 0.26 MB (hbm) | ✓ 0.00026×tol | 0.36x vs nitrix-jax |
| laplacian | jax-cpu | n=512 | `cupy.laplacian` | skipped | — | — | — | — | — |
| laplacian | jax-cpu | n=512 | `nitrix-jax` | ok | 149.8 µs / 174.2 µs | 115.60 ms | 704 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cpu | n=512 | `scipy.csgraph.laplacian` | ok | 98.4 µs / 106.1 µs | 129.5 µs | 704 MB (rss) | ✓ 0.00054×tol | 0.66x vs nitrix-jax |
| laplacian | jax-cuda12 | n=512 | `cupy.laplacian` | ok | 72.4 µs / 77.8 µs | 60.23 ms | 1.05 MB (hbm) | ✓ 0.00014×tol | 0.58x vs nitrix-jax |
| laplacian | jax-cuda12 | n=512 | `nitrix-jax` | ok | 123.9 µs / 130.0 µs | 164.42 ms | 3.15 MB (hbm) | ✓ 0.00054×tol | 1.00x vs nitrix-jax |
| laplacian | jax-cuda12 | n=512 | `scipy.csgraph.laplacian` | ok | 99.0 µs / 120.5 µs | 132.8 µs | 1.05 MB (hbm) | ✓ 0.00054×tol | 0.80x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

