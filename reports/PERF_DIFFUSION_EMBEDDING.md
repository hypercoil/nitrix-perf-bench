# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c31eaa10bdfaa772b7f68229556aabf5ebc9ec33 | bench: fff6ce2ada36ae6e981b89cb575e55a3090bed31
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T17:15:29.987212+00:00

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
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 30.44 ms / 32.52 ms | 2.187 s | 4.19 MB (hbm) | ✓ 9.7e-05×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 32.90 ms / 33.28 ms | 3.009 s | 138.49 MB (hbm) | ✓ 0.012×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 30.88 ms / 31.00 ms | 3.929 s | 138.49 MB (hbm) | ✓ 0.018×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-poly` | ok | 16.43 ms / 16.50 ms | 3.783 s | 138.49 MB (hbm) | ≈ 15×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 22.80 ms / 22.96 ms | 3.550 s | 155.37 MB (hbm) | ≈ 30×tol | — |
| diffusion_embedding | jax-cuda12 | n=1024,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 46.58 ms / 47.65 ms | 48.56 ms | 4.19 MB (hbm) | ✓ 1.4e-11×tol | — |
| diffusion_embedding | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 129.82 ms / 154.29 ms | 1.106 s | 8.39 MB (hbm) | n/a (no oracle) | 3.09x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 42.00 ms / 42.13 ms | 3.411 s | 143.34 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 41.89 ms / 42.08 ms | 3.819 s | 143.34 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=10242,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | ok | 317.39 ms / 345.10 ms | 346.40 ms | 8.39 MB (hbm) | n/a (no oracle) | 7.56x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 374.97 ms / 410.68 ms | 854.99 ms | 67.11 MB (hbm) | n/a (no oracle) | 8.10x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 46.31 ms / 46.68 ms | 3.759 s | 204.47 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 47.88 ms / 48.68 ms | 4.144 s | 204.47 MB (hbm) | n/a (no oracle) | 1.03x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=120000,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | timeout | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax` | ok | 530.86 ms / 542.73 ms | 629.08 ms | 918 MB (rss) | ✓ 0.00096×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 174.12 ms / 177.75 ms | 1.651 s | 918 MB (rss) | ✓ 0.14×tol | 0.33x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 260.90 ms / 280.20 ms | 1.734 s | 918 MB (rss) | ✓ 0.13×tol | 0.49x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 231.07 ms / 234.25 ms | 1.747 s | 918 MB (rss) | ≈ 12×tol | 0.44x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 320.42 ms / 324.71 ms | 1.869 s | 918 MB (rss) | ≈ 33×tol | 0.60x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 269.31 ms / 290.15 ms | 268.53 ms | 918 MB (rss) | ✓ 2e-11×tol | 0.51x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `cupyx.sparse.eigsh` | ok | 44.67 ms / 49.07 ms | 553.82 ms | 16.78 MB (hbm) | ✓ 0.0003×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg` | ok | 36.45 ms / 36.79 ms | 3.248 s | 117.44 MB (hbm) | ✓ 0.077×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-lobpcg-vjp` | ok | 36.05 ms / 36.46 ms | 4.011 s | 151.03 MB (hbm) | ✓ 0.069×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-poly` | ok | 19.43 ms / 19.51 ms | 3.737 s | 117.96 MB (hbm) | ≈ 12×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `nitrix-jax-shift_invert` | ok | 20.92 ms / 21.02 ms | 3.839 s | 151.14 MB (hbm) | ≈ 33×tol | — |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=dense | `scipy.sparse.eigsh` | ok | 249.84 ms / 276.40 ms | 511.84 ms | 16.78 MB (hbm) | ✓ 3.7e-11×tol | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | skipped | — | — | — | — | — |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 242.60 ms / 245.13 ms | 1.566 s | 918 MB (rss) | ✓ 0.14×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 230.99 ms / 233.09 ms | 1.683 s | 918 MB (rss) | ✓ 0.13×tol | 0.95x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 353.70 ms / 358.58 ms | 1.831 s | 918 MB (rss) | ≈ 12×tol | 1.46x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 481.53 ms / 488.30 ms | 2.152 s | 918 MB (rss) | ≈ 33×tol | 1.98x vs nitrix-jax |
| diffusion_embedding | jax-cpu | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 259.53 ms / 270.53 ms | 259.38 ms | 918 MB (rss) | ✓ 7.7e-12×tol | 1.07x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `cupyx.sparse.eigsh` | ok | 44.71 ms / 49.64 ms | 453.58 ms | 20.15 MB (hbm) | ✓ 0.00037×tol | 0.46x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax` | ok | 97.00 ms / 99.37 ms | 2.980 s | 120.96 MB (hbm) | ✓ 0.064×tol | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-lobpcg-vjp` | ok | 94.82 ms / 95.30 ms | 3.692 s | 120.96 MB (hbm) | ✓ 0.1×tol | 0.98x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-poly` | ok | 128.85 ms / 131.90 ms | 3.462 s | 120.96 MB (hbm) | ≈ 11×tol | 1.33x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `nitrix-jax-shift_invert` | ok | 183.07 ms / 185.40 ms | 3.750 s | 104.19 MB (hbm) | ≈ 33×tol | 1.89x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=2048,k=8,fmt=ell | `scipy.sparse.eigsh` | ok | 243.57 ms / 263.60 ms | 257.37 ms | 20.15 MB (hbm) | ✓ 1.5e-11×tol | 2.51x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `cupyx.sparse.eigsh` | ok | 202.86 ms / 234.61 ms | 736.11 ms | 33.55 MB (hbm) | n/a (no oracle) | 8.92x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax` | ok | 22.73 ms / 22.77 ms | 2.871 s | 153.94 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `nitrix-jax-lobpcg-vjp` | ok | 23.10 ms / 23.24 ms | 3.338 s | 153.94 MB (hbm) | n/a (no oracle) | 1.02x vs nitrix-jax |
| diffusion_embedding | jax-cuda12 | n=40962,degree=16,k=8,fmt=ell,tier=large | `scipy.sparse.eigsh` | ok | 1.975 s / 2.394 s | 2.250 s | 33.55 MB (hbm) | n/a (no oracle) | 86.87x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

