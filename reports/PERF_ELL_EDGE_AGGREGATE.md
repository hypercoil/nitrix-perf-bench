# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c90f6eae1488d27e013d67260d9946fbfadcdbb3 | bench: 5277d79a37c288dbf5fa81ceb8fa30416dfe038c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-23T20:46:24.889416+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cuda12** — NVIDIA A10G (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| ell_edge_aggregate | jax-cpu | n=16384,k=16,d_in=64,d_out=64,algebra=real | `nitrix-jax` | ok | 102.01 ms / 110.62 ms | 235.29 ms | 470 MB (rss) | ✓ 0.027×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=16384,k=16,d_in=64,d_out=64,algebra=real | `pyg` | ok | 54.09 ms / 57.99 ms | 1.040 s | 592 MB (rss) | ✓ 0.029×tol | 0.53x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=16384,k=16,d_in=64,d_out=64,algebra=real | `nitrix-jax` | ok | 825.1 µs / 841.4 µs | 643.89 ms | 312.49 MB (hbm) | ✓ 0.029×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=16384,k=16,d_in=64,d_out=64,algebra=real | `pyg` | ok | 778.1 µs / 784.1 µs | 1.332 s | 164.77 MB (hbm) | ✓ 0.029×tol | 0.94x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=16384,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `nitrix-jax` | ok | 93.38 ms / 102.78 ms | 234.97 ms | 470 MB (rss) | ✓ 0.0022×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=16384,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `pyg` | ok | 52.98 ms / 54.81 ms | 1.010 s | 590 MB (rss) | ✓ 0.0022×tol | 0.57x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=16384,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `nitrix-jax` | ok | 825.6 µs / 856.1 µs | 509.68 ms | 312.49 MB (hbm) | ✓ 0.0022×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=16384,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `pyg` | ok | 947.0 µs / 971.2 µs | 1.578 s | 164.77 MB (hbm) | ✓ 0.0022×tol | 1.15x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=4096,k=16,d_in=64,d_out=64,algebra=real | `nitrix-jax` | ok | 18.68 ms / 22.95 ms | 127.82 ms | 470 MB (rss) | ✓ 0.019×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=4096,k=16,d_in=64,d_out=64,algebra=real | `pyg` | ok | 4.29 ms / 9.74 ms | 976.94 ms | 470 MB (rss) | ✓ 0.019×tol | 0.23x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=4096,k=16,d_in=64,d_out=64,algebra=real | `nitrix-jax` | ok | 297.9 µs / 303.2 µs | 363.07 ms | 106.45 MB (hbm) | ✓ 0.02×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=4096,k=16,d_in=64,d_out=64,algebra=real | `pyg` | ok | 283.8 µs / 291.8 µs | 1.296 s | 47.60 MB (hbm) | ✓ 0.019×tol | 0.95x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=4096,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `nitrix-jax` | ok | 19.53 ms / 22.50 ms | 131.04 ms | 470 MB (rss) | ✓ 0.0024×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cpu | n=4096,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `pyg` | ok | 4.18 ms / 9.58 ms | 1.006 s | 470 MB (rss) | ✓ 0.0024×tol | 0.21x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=4096,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `nitrix-jax` | ok | 280.3 µs / 297.8 µs | 369.18 ms | 106.45 MB (hbm) | ✓ 0.0024×tol | 1.00x vs nitrix-jax |
| ell_edge_aggregate | jax-cuda12 | n=4096,k=16,d_in=64,d_out=64,algebra=tropical_max_plus | `pyg` | ok | 318.7 µs / 328.0 µs | 1.425 s | 47.60 MB (hbm) | ✓ 0.0024×tol | 1.14x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

