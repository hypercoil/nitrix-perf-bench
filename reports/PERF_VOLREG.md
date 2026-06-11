# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 159777581af22f900e3de1a4c3446ad168444ed4 | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T00:10:05.124655+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | ok | 1.012 s / 1.057 s | 7.026 s | 1633 MB (rss) | n/a (no oracle) | 15.63x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 64.72 ms / 69.99 ms | 985.99 ms | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 1.42 ms / 1.45 ms | 2.896 s | 136.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | ok | 2.013 s / 2.767 s | 6.909 s | 1633 MB (rss) | n/a (no oracle) | 15.44x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 130.40 ms / 135.48 ms | 886.60 ms | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 3.06 ms / 3.10 ms | 2.876 s | 138.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | ok | 599.07 ms / 743.80 ms | 9.777 s | 1633 MB (rss) | n/a (no oracle) | 21.34x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 28.07 ms / 28.52 ms | 779.04 ms | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 988.9 µs / 1.04 ms | 3.171 s | 135.53 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 10.086 s / 10.191 s | 14.867 s | 1633 MB (rss) | n/a (no oracle) | 2.41x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 4.185 s / 4.435 s | 7.042 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 172.66 ms / 172.69 ms | 8.045 s | 625.34 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `ants.motion_correction` | ok | 23.586 s / 23.626 s | 28.066 s | 1633 MB (rss) | n/a (no oracle) | 3.09x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=200,levels=2,iters=20 | `nitrix-jax` | ok | 7.621 s / 7.959 s | 11.175 s | 1711 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=200,levels=2,iters=20 | `nitrix-jax` | ok | 355.52 ms / 355.97 ms | 8.135 s | 1207.97 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `ants.motion_correction` | ok | 4.390 s / 4.513 s | 8.031 s | 1633 MB (rss) | n/a (no oracle) | 2.31x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=50,levels=2,iters=20 | `nitrix-jax` | ok | 1.897 s / 2.136 s | 5.020 s | 1633 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=50,levels=2,iters=20 | `nitrix-jax` | ok | 81.21 ms / 81.35 ms | 8.240 s | 334.79 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `ants.motion_correction` | ok | 85.096 s / 100.870 s | 79.926 s | 2100 MB (rss) | n/a (no oracle) | 3.92x vs nitrix-jax |
| volreg | jax-cpu | shape=[48, 48, 48],T=500,levels=2,iters=20 | `nitrix-jax` | ok | 21.732 s / 22.507 s | 24.723 s | 3072 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[48, 48, 48],T=500,levels=2,iters=20 | `nitrix-jax` | ok | 904.54 ms / 904.68 ms | 9.886 s | 2815.25 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 24.101 s / 24.615 s | 29.375 s | 1633 MB (rss) | n/a (no oracle) | 2.22x vs nitrix-jax |
| volreg | jax-cpu | shape=[64, 64, 64],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 10.844 s / 11.812 s | 13.710 s | 1812 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[64, 64, 64],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 443.42 ms / 443.54 ms | 9.378 s | 1347.45 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `ants.motion_correction` | ok | 29.684 s / 30.682 s | 37.960 s | 1999 MB (rss) | n/a (no oracle) | 1.01x vs nitrix-jax |
| volreg | jax-cpu | shape=[80, 80, 80],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 29.307 s / 29.575 s | 32.876 s | 3053 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[80, 80, 80],T=100,levels=2,iters=20 | `nitrix-jax` | ok | 1.088 s / 1.090 s | 10.894 s | 2638.00 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

