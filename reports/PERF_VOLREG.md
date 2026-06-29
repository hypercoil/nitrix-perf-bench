# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T21:28:17.542266+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.3dvolreg` | ok | 783.29 ms / 807.26 ms | 1.063 s | 485 MB (rss) | n/a (no oracle) | 2.73x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.iofloor` | ok | 311.26 ms / 319.46 ms | 441.69 ms | 485 MB (rss) | n/a (no oracle) | 1.08x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | ok | 19.726 s / 20.053 s | 40.280 s | 478 MB (rss) | n/a (no oracle) | 68.74x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.iofloor` | ok | 346.62 ms / 357.83 ms | 510.88 ms | 484 MB (rss) | n/a (no oracle) | 1.21x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.mcflirt` | ok | 494.91 ms / 508.15 ms | 654.61 ms | 484 MB (rss) | n/a (no oracle) | 1.72x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 286.99 ms / 299.19 ms | 1.343 s | 643 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=16,levels=1,iters=10 | `nitrix-jax` | ok | 1.45 ms / 1.46 ms | 7.279 s | 136.32 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.3dvolreg` | ok | 1.461 s / 1.477 s | 1.655 s | 499 MB (rss) | n/a (no oracle) | 2.27x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.iofloor` | ok | 477.47 ms / 506.18 ms | 641.22 ms | 498 MB (rss) | n/a (no oracle) | 0.74x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | ok | 39.934 s / 50.023 s | 49.667 s | 478 MB (rss) | n/a (no oracle) | 62.17x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.iofloor` | ok | 544.36 ms / 564.81 ms | 691.38 ms | 498 MB (rss) | n/a (no oracle) | 0.85x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.mcflirt` | ok | 803.37 ms / 832.11 ms | 965.55 ms | 498 MB (rss) | n/a (no oracle) | 1.25x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 642.38 ms / 648.49 ms | 1.710 s | 680 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=32,levels=1,iters=10 | `nitrix-jax` | ok | 2.22 ms / 2.25 ms | 5.382 s | 138.41 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.3dvolreg` | ok | 495.02 ms / 503.01 ms | 4.776 s | 478 MB (rss) | n/a (no oracle) | 4.16x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.iofloor` | ok | 202.37 ms / 214.03 ms | 389.01 ms | 478 MB (rss) | n/a (no oracle) | 1.70x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | ok | 10.137 s / 11.236 s | 14.182 s | 478 MB (rss) | n/a (no oracle) | 85.19x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.iofloor` | ok | 234.71 ms / 242.52 ms | 370.13 ms | 478 MB (rss) | n/a (no oracle) | 1.97x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.mcflirt` | ok | 329.29 ms / 338.19 ms | 591.52 ms | 478 MB (rss) | n/a (no oracle) | 2.77x vs nitrix-jax |
| volreg | jax-cpu | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 118.99 ms / 121.17 ms | 1.076 s | 639 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.3dvolreg` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `afni.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `ants.motion_correction` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.iofloor` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `fsl.mcflirt` | skipped | — | — | — | — | — |
| volreg | jax-cuda12 | shape=[32, 32, 32],T=8,levels=1,iters=10 | `nitrix-jax` | ok | 1.03 ms / 1.05 ms | 4.348 s | 135.53 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

