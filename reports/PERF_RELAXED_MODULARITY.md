# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 32fd5ab9420d25d8be13008bc3b162856e0fcad7 | bench: 8157e5089aaf5c007d82fc83adc5ffc5a9c2874c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T21:27:48.722495+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| relaxed_modularity | jax-cpu | n=128 | `cupy.relaxed_modularity` | skipped | — | — | — | — | — |
| relaxed_modularity | jax-cpu | n=128 | `networkx.modularity` | ok | 2.79 ms / 2.90 ms | 732.96 ms | 706 MB (rss) | ✓ 3.8e-14×tol | 63.95x vs nitrix-jax |
| relaxed_modularity | jax-cpu | n=128 | `nitrix-jax` | ok | 43.6 µs / 45.3 µs | 152.56 ms | 706 MB (rss) | ✓ 5.6e-05×tol | 1.00x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=128 | `cupy.relaxed_modularity` | ok | 223.6 µs / 234.1 µs | 358.80 ms | 0.07 MB (hbm) | ✓ 2.1e-05×tol | 1.88x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=128 | `networkx.modularity` | ok | 3.10 ms / 4.36 ms | 508.84 ms | 0.07 MB (hbm) | ✓ 3.8e-14×tol | 26.11x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=128 | `nitrix-jax` | ok | 118.8 µs / 139.3 µs | 552.92 ms | 71.50 MB (hbm) | ✓ 2.4e-05×tol | 1.00x vs nitrix-jax |
| relaxed_modularity | jax-cpu | n=256 | `cupy.relaxed_modularity` | skipped | — | — | — | — | — |
| relaxed_modularity | jax-cpu | n=256 | `networkx.modularity` | ok | 12.75 ms / 15.90 ms | 569.98 ms | 706 MB (rss) | ✓ 1.1e-13×tol | 91.70x vs nitrix-jax |
| relaxed_modularity | jax-cpu | n=256 | `nitrix-jax` | ok | 139.1 µs / 194.0 µs | 205.36 ms | 706 MB (rss) | ✓ 6.2e-05×tol | 1.00x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=256 | `cupy.relaxed_modularity` | ok | 227.6 µs / 239.1 µs | 271.95 ms | 0.27 MB (hbm) | ✓ 5e-05×tol | 1.56x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=256 | `networkx.modularity` | ok | 11.12 ms / 12.96 ms | 518.56 ms | 0.27 MB (hbm) | ✓ 1.1e-13×tol | 76.40x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=256 | `nitrix-jax` | ok | 145.5 µs / 147.8 µs | 726.29 ms | 134.49 MB (hbm) | ✓ 4.3e-06×tol | 1.00x vs nitrix-jax |
| relaxed_modularity | jax-cpu | n=512 | `cupy.relaxed_modularity` | skipped | — | — | — | — | — |
| relaxed_modularity | jax-cpu | n=512 | `networkx.modularity` | ok | 53.28 ms / 65.59 ms | 554.99 ms | 706 MB (rss) | ✓ 3e-14×tol | 192.14x vs nitrix-jax |
| relaxed_modularity | jax-cpu | n=512 | `nitrix-jax` | ok | 277.3 µs / 365.1 µs | 199.70 ms | 706 MB (rss) | ✓ 7e-06×tol | 1.00x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=512 | `cupy.relaxed_modularity` | ok | 222.3 µs / 244.9 µs | 265.86 ms | 1.06 MB (hbm) | ✓ 1.4e-05×tol | 1.83x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=512 | `networkx.modularity` | ok | 45.35 ms / 48.64 ms | 548.32 ms | 1.06 MB (hbm) | ✓ 3e-14×tol | 373.56x vs nitrix-jax |
| relaxed_modularity | jax-cuda12 | n=512 | `nitrix-jax` | ok | 121.4 µs / 134.9 µs | 771.25 ms | 135.29 MB (hbm) | ✓ 1.8e-05×tol | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

