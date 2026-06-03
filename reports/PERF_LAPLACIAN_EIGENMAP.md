# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 88a23053af93d9466c3993dae0309eddd5c11c6f | bench: 86415b9f30cd34c38c2c4030b6a84d1b896d493c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T05:03:57.628648+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| laplacian_eigenmap | jax-cpu | n=1024 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=1024 | `nitrix-jax` | ok | 355.09 ms / 360.89 ms | 1.935 s | 855 MB (rss) | ✓ 8.6e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024 | `nitrix-jax-eigh` | ok | 153.38 ms / 283.96 ms | 3.765 s | 855 MB (rss) | ✓ 4.2e-06×tol | 0.43x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=1024 | `scipy.sparse.eigsh` | ok | 58.11 ms / 60.79 ms | 60.75 ms | 855 MB (rss) | ✓ 8e-13×tol | 0.16x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024 | `cupyx.sparse.eigsh` | ok | 45.15 ms / 52.29 ms | 343.72 ms | 4.19 MB (hbm) | ✓ 6.8e-06×tol | 0.08x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024 | `nitrix-jax` | ok | 551.74 ms / 552.64 ms | 5.156 s | 104.86 MB (hbm) | ✓ 3.6e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=1024 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=1024 | `scipy.sparse.eigsh` | ok | 44.02 ms / 50.17 ms | 44.78 ms | 4.19 MB (hbm) | ✓ 3.4e-13×tol | 0.08x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=2048 | `nitrix-jax` | ok | 1.122 s / 1.129 s | 2.676 s | 855 MB (rss) | ✓ 2.4e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048 | `nitrix-jax-eigh` | ok | 891.61 ms / 2.042 s | 1.733 s | 855 MB (rss) | ✓ 6.6e-06×tol | 0.79x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=2048 | `scipy.sparse.eigsh` | ok | 242.72 ms / 251.87 ms | 223.53 ms | 855 MB (rss) | ✓ 8.6e-13×tol | 0.22x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048 | `cupyx.sparse.eigsh` | ok | 60.23 ms / 81.75 ms | 411.91 ms | 16.78 MB (hbm) | ✓ 3.8e-06×tol | 0.06x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 1.001 s / 1.189 s | 5.696 s | 123.40 MB (hbm) | ✓ 2.8e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=2048 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=2048 | `scipy.sparse.eigsh` | ok | 245.02 ms / 250.61 ms | 240.09 ms | 16.78 MB (hbm) | ✓ 6.4e-13×tol | 0.24x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=512 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cpu | n=512 | `nitrix-jax` | ok | 147.08 ms / 148.95 ms | 1.442 s | 855 MB (rss) | ✓ 7.1e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=512 | `nitrix-jax-eigh` | ok | 17.17 ms / 53.84 ms | 324.00 ms | 855 MB (rss) | ✓ 8.2e-06×tol | 0.12x vs nitrix-jax |
| laplacian_eigenmap | jax-cpu | n=512 | `scipy.sparse.eigsh` | ok | 16.71 ms / 17.26 ms | 21.39 ms | 855 MB (rss) | ✓ 5.2e-13×tol | 0.11x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=512 | `cupyx.sparse.eigsh` | ok | 43.56 ms / 57.44 ms | 2.250 s | 1.05 MB (hbm) | ✓ 4.5e-06×tol | 0.12x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=512 | `nitrix-jax` | ok | 360.77 ms / 361.12 ms | 3.958 s | 101.71 MB (hbm) | ✓ 5.2e-07×tol | 1.00x vs nitrix-jax |
| laplacian_eigenmap | jax-cuda12 | n=512 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| laplacian_eigenmap | jax-cuda12 | n=512 | `scipy.sparse.eigsh` | ok | 14.58 ms / 15.79 ms | 18.79 ms | 1.05 MB (hbm) | ✓ 4e-13×tol | 0.04x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

