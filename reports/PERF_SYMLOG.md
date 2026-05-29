# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: daec36e9ca79a88bd13028edca7a0a02eebbfc7e
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T04:06:27.893481+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| symlog | jax-cpu | d=256 | `cupy.eigh_logm` | skipped | — | — | — | — | — |
| symlog | jax-cpu | d=256 | `nitrix-jax` | ok | 5.05 ms / 5.43 ms | 328.68 ms | 522 MB (rss) | ✓ 0.00048×tol | 1.00x vs nitrix-jax |
| symlog | jax-cpu | d=256 | `scipy.linalg.logm` | ok | 250.83 ms / 358.58 ms | 520.65 ms | 473 MB (rss) | ✓ 0.0012×tol | 49.65x vs nitrix-jax |
| symlog | jax-cuda12 | d=256 | `cupy.eigh_logm` | skipped | — | — | — | — | — |
| symlog | jax-cuda12 | d=256 | `nitrix-jax` | ok | 1.95 ms / 1.96 ms | 1.214 s | 72.09 MB (hbm) | ✓ 0.00064×tol | 1.00x vs nitrix-jax |
| symlog | jax-cuda12 | d=256 | `scipy.linalg.logm` | ok | 170.22 ms / 298.03 ms | 1.602 s | 0.26 MB (hbm) | ✓ 0.0012×tol | 87.32x vs nitrix-jax |
| symlog | jax-cpu | d=512 | `cupy.eigh_logm` | skipped | — | — | — | — | — |
| symlog | jax-cpu | d=512 | `nitrix-jax` | ok | 25.64 ms / 30.55 ms | 2.215 s | 550 MB (rss) | ✓ 0.00057×tol | 1.00x vs nitrix-jax |
| symlog | jax-cpu | d=512 | `scipy.linalg.logm` | ok | 676.01 ms / 2.582 s | 1.373 s | 500 MB (rss) | ✓ 0.0014×tol | 26.37x vs nitrix-jax |
| symlog | jax-cuda12 | d=512 | `cupy.eigh_logm` | skipped | — | — | — | — | — |
| symlog | jax-cuda12 | d=512 | `nitrix-jax` | ok | 4.29 ms / 4.31 ms | 1.208 s | 74.45 MB (hbm) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| symlog | jax-cuda12 | d=512 | `scipy.linalg.logm` | ok | 567.46 ms / 1.086 s | 4.004 s | 1.05 MB (hbm) | ✓ 0.0014×tol | 132.39x vs nitrix-jax |
| symlog | jax-cpu | d=64 | `cupy.eigh_logm` | skipped | — | — | — | — | — |
| symlog | jax-cpu | d=64 | `nitrix-jax` | ok | 387.4 µs / 397.0 µs | 172.98 ms | 516 MB (rss) | ✓ 0.00044×tol | 1.00x vs nitrix-jax |
| symlog | jax-cpu | d=64 | `scipy.linalg.logm` | ok | 4.64 ms / 5.00 ms | 174.22 ms | 471 MB (rss) | ✓ 0.00086×tol | 11.97x vs nitrix-jax |
| symlog | jax-cuda12 | d=64 | `cupy.eigh_logm` | ok | 705.9 µs / 709.9 µs | 1.118 s | 0.02 MB (hbm) | ✓ 0.00048×tol | 0.95x vs nitrix-jax |
| symlog | jax-cuda12 | d=64 | `nitrix-jax` | ok | 746.3 µs / 751.7 µs | 952.87 ms | 71.35 MB (hbm) | ✓ 0.00048×tol | 1.00x vs nitrix-jax |
| symlog | jax-cuda12 | d=64 | `scipy.linalg.logm` | ok | 64.42 ms / 110.01 ms | 260.89 ms | 0.02 MB (hbm) | ✓ 0.00086×tol | 86.31x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

