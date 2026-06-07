# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 11f91e38159ffe403128b6cf08cf52be19798870 | bench: 14b8ae1fea1eae07a45ad92503d9604a2be294a4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T22:58:30.601157+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax` | ok | 93.50 ms / 100.72 ms | 233.36 ms | 918 MB (rss) | ✓ 0.00054×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 54.75 ms / 55.18 ms | 1.271 s | 918 MB (rss) | ✓ 0.025×tol | 0.59x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 68.38 ms / 71.84 ms | 1.480 s | 918 MB (rss) | ✓ 0.035×tol | 0.73x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 63.50 ms / 65.77 ms | 1.383 s | 918 MB (rss) | ≈ 17×tol | 0.68x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 126.47 ms / 127.66 ms | 1.650 s | 918 MB (rss) | ≈ 30×tol | 1.35x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 47.26 ms / 48.28 ms | 51.06 ms | 918 MB (rss) | ✓ 1.4e-11×tol | 0.51x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 54.71 ms / 89.06 ms | 459.65 ms | 4.19 MB (hbm) | ✓ 9.7e-05×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 30.71 ms / 31.28 ms | 3.616 s | 138.49 MB (hbm) | ✓ 0.02×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 30.54 ms / 30.67 ms | 5.073 s | 138.49 MB (hbm) | ✓ 0.016×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 17.61 ms / 19.70 ms | 4.603 s | 138.49 MB (hbm) | ≈ 19×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 23.02 ms / 24.84 ms | 4.204 s | 155.37 MB (hbm) | ≈ 30×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 66.98 ms / 85.67 ms | 97.35 ms | 4.19 MB (hbm) | ✓ 1.3e-11×tol | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax` | ok | 530.86 ms / 542.73 ms | 629.08 ms | 918 MB (rss) | ✓ 0.00096×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 174.12 ms / 177.75 ms | 1.651 s | 918 MB (rss) | ✓ 0.14×tol | 0.33x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 260.90 ms / 280.20 ms | 1.734 s | 918 MB (rss) | ✓ 0.13×tol | 0.49x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 231.07 ms / 234.25 ms | 1.747 s | 918 MB (rss) | ≈ 12×tol | 0.44x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 320.42 ms / 324.71 ms | 1.869 s | 918 MB (rss) | ≈ 33×tol | 0.60x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 269.31 ms / 290.15 ms | 268.53 ms | 918 MB (rss) | ✓ 2e-11×tol | 0.51x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 117.16 ms / 309.15 ms | 581.64 ms | 16.78 MB (hbm) | ✓ 0.00014×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 37.70 ms / 39.83 ms | 3.705 s | 117.44 MB (hbm) | ✓ 0.061×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 35.70 ms / 36.71 ms | 4.711 s | 151.03 MB (hbm) | ✓ 0.081×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 20.90 ms / 22.46 ms | 4.806 s | 117.96 MB (hbm) | ≈ 12×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 23.22 ms / 23.90 ms | 5.458 s | 151.14 MB (hbm) | ≈ 33×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 556.56 ms / 668.77 ms | 548.16 ms | 16.78 MB (hbm) | ✓ 1.4e-11×tol | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 242.60 ms / 245.13 ms | 1.566 s | 918 MB (rss) | ✓ 0.14×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 230.99 ms / 233.09 ms | 1.683 s | 918 MB (rss) | ✓ 0.13×tol | 0.95x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 353.70 ms / 358.58 ms | 1.831 s | 918 MB (rss) | ≈ 12×tol | 1.46x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 481.53 ms / 488.30 ms | 2.152 s | 918 MB (rss) | ≈ 33×tol | 1.98x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 259.53 ms / 270.53 ms | 259.38 ms | 918 MB (rss) | ✓ 7.7e-12×tol | 1.07x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 68.69 ms / 93.14 ms | 496.27 ms | 20.15 MB (hbm) | ✓ 0.00013×tol | 0.64x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 107.91 ms / 116.15 ms | 3.522 s | 120.96 MB (hbm) | ✓ 0.059×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 107.44 ms / 121.24 ms | 4.380 s | 120.96 MB (hbm) | ✓ 0.056×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 137.61 ms / 165.53 ms | 4.037 s | 120.96 MB (hbm) | ≈ 11×tol | 1.28x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 191.43 ms / 242.70 ms | 4.752 s | 104.19 MB (hbm) | ≈ 33×tol | 1.77x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 432.54 ms / 519.71 ms | 483.91 ms | 20.15 MB (hbm) | ✓ 1.2e-11×tol | 4.01x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

