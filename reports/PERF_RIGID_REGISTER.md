# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fbd2597242ee136b27625b179cd025d81183a95d | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T00:35:23.278210+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| rigid_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 1.346 s / 1.395 s | 3.592 s | 857 MB (rss) | n/a (no oracle) | 0.24x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 5.683 s / 5.689 s | 8.359 s | 1331 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 227.88 ms / 229.95 ms | 12.220 s | 8741.46 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 500.88 ms / 504.27 ms | 5.287 s | 857 MB (rss) | n/a (no oracle) | 0.09x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 5.658 s / 5.723 s | 9.388 s | 1267 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 231.74 ms / 232.51 ms | 20.121 s | 8665.56 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 777.03 ms / 825.89 ms | 6.411 s | 857 MB (rss) | n/a (no oracle) | 0.06x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 12.004 s / 12.326 s | 14.281 s | 1958 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 415.41 ms / 415.96 ms | 11.276 s | 13694.15 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | ok | 1.016 s / 1.019 s | 5.239 s | 868 MB (rss) | n/a (no oracle) | 0.05x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 20.578 s / 20.808 s | 23.433 s | 2914 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 808.46 ms / 809.96 ms | 21.258 s | 1832.91 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 196.57 ms / 209.28 ms | 5.298 s | 857 MB (rss) | n/a (no oracle) | 0.89x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 221.93 ms / 223.88 ms | 1.537 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 5.02 ms / 5.02 ms | 3.701 s | 151.88 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 247.38 ms / 249.97 ms | 3.559 s | 857 MB (rss) | n/a (no oracle) | 0.63x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 392.03 ms / 445.87 ms | 4.211 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 14.95 ms / 14.98 ms | 10.169 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 257.39 ms / 284.11 ms | 5.060 s | 857 MB (rss) | n/a (no oracle) | 0.64x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 402.35 ms / 618.32 ms | 6.225 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 26.37 ms / 50.13 ms | 12.208 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 344.60 ms / 414.67 ms | 5.664 s | 857 MB (rss) | n/a (no oracle) | 0.13x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 2.664 s / 2.704 s | 7.406 s | 1022 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 91.84 ms / 92.02 ms | 9.168 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 305.09 ms / 357.24 ms | 4.874 s | 857 MB (rss) | n/a (no oracle) | 0.13x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 2.411 s / 2.460 s | 6.764 s | 1026 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 101.31 ms / 158.66 ms | 12.210 s | 1514.57 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

