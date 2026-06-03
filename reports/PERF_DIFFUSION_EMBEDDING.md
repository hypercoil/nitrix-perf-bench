# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 32fd5ab9420d25d8be13008bc3b162856e0fcad7 | bench: cad0d40aa442b1582b825904fc1d43bcf23be223
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T07:09:14.633493+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_embedding | jax-cpu | n=1024 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=1024 | `nitrix-jax` | ok | 361.58 ms / 367.98 ms | 2.053 s | 879 MB (rss) | ✓ 1.2e-06×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024 | `nitrix-jax-eigh` | ok | 224.09 ms / 438.43 ms | 474.64 ms | 879 MB (rss) | ✓ 9.4e-06×tol | 0.62x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024 | `scipy.sparse.eigsh` | ok | 60.78 ms / 69.39 ms | 68.25 ms | 879 MB (rss) | ✓ 2.4e-13×tol | 0.17x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=1024 | `cupyx.sparse.eigsh` | ok | 43.94 ms / 44.84 ms | 360.05 ms | 4.19 MB (hbm) | ✓ 2.6e-06×tol | 0.08x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=1024 | `nitrix-jax` | ok | 528.96 ms / 531.10 ms | 5.317 s | 104.86 MB (hbm) | ✓ 7.7e-07×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=1024 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=1024 | `scipy.sparse.eigsh` | ok | 47.71 ms / 51.99 ms | 50.20 ms | 4.19 MB (hbm) | ✓ 1.5e-13×tol | 0.09x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048 | `nitrix-jax` | ok | 1.127 s / 1.136 s | 2.875 s | 879 MB (rss) | ✓ 6.6e-07×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048 | `nitrix-jax-eigh` | ok | 902.00 ms / 1.864 s | 2.427 s | 879 MB (rss) | ✓ 1.5e-05×tol | 0.80x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048 | `scipy.sparse.eigsh` | ok | 251.95 ms / 285.43 ms | 295.50 ms | 879 MB (rss) | ✓ 3.1e-13×tol | 0.22x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048 | `cupyx.sparse.eigsh` | ok | 73.96 ms / 84.78 ms | 453.45 ms | 16.78 MB (hbm) | ✓ 2.4e-06×tol | 0.06x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 1.164 s / 1.251 s | 5.978 s | 117.44 MB (hbm) | ✓ 3.8e-07×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=2048 | `scipy.sparse.eigsh` | ok | 235.55 ms / 251.75 ms | 387.14 ms | 16.78 MB (hbm) | ✓ 2.4e-13×tol | 0.20x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=512 | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=512 | `nitrix-jax` | ok | 163.00 ms / 168.62 ms | 2.272 s | 879 MB (rss) | ✓ 2e-06×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=512 | `nitrix-jax-eigh` | ok | 23.69 ms / 36.85 ms | 283.10 ms | 879 MB (rss) | ✓ 1.1e-05×tol | 0.15x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=512 | `scipy.sparse.eigsh` | ok | 16.68 ms / 17.30 ms | 27.63 ms | 879 MB (rss) | ✓ 2.5e-13×tol | 0.10x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=512 | `cupyx.sparse.eigsh` | ok | 36.01 ms / 36.80 ms | 563.44 ms | 1.05 MB (hbm) | ✓ 3.9e-06×tol | 0.10x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=512 | `nitrix-jax` | ok | 349.66 ms / 361.08 ms | 4.106 s | 101.71 MB (hbm) | ✓ 1.2e-06×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=512 | `nitrix-jax-eigh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=512 | `scipy.sparse.eigsh` | ok | 15.76 ms / 16.69 ms | 20.41 ms | 1.05 MB (hbm) | ✓ 3.1e-13×tol | 0.05x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

