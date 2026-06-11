# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 9e53019e8fa652aaa379a09ac190bb18c0d8e3a8 | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T00:42:25.550236+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | ok | 727.04 ms / 731.90 ms | 9.312 s | 857 MB (rss) | n/a (no oracle) | 0.07x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 10.079 s / 10.316 s | 13.712 s | 1661 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=20 | `nitrix-jax` | ok | 302.79 ms / 306.03 ms | 8.906 s | 8741.46 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 863.08 ms / 870.02 ms | 3.751 s | 857 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 10.831 s / 13.466 s | 15.379 s | 1672 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[128, 128, 128],moving_shape=[112, 128, 144],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 312.50 ms / 317.24 ms | 31.286 s | 8665.56 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | ok | 1.002 s / 1.090 s | 5.452 s | 857 MB (rss) | n/a (no oracle) | 0.05x vs nitrix-jax |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 21.276 s / 21.390 s | 26.154 s | 2554 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[160, 160, 160],levels=2,iters=20 | `nitrix-jax` | ok | 556.18 ms / 557.26 ms | 22.612 s | 1924.66 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | ok | 1.395 s / 1.522 s | 7.232 s | 870 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 37.959 s / 38.608 s | 46.074 s | 3761 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[192, 192, 192],levels=2,iters=20 | `nitrix-jax` | ok | 1.036 s / 1.036 s | 24.724 s | 3356.49 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 262.46 ms / 326.03 ms | 4.216 s | 857 MB (rss) | n/a (no oracle) | 1.96x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 133.96 ms / 134.78 ms | 1.542 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 7.52 ms / 7.53 ms | 3.052 s | 151.88 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 213.54 ms / 213.94 ms | 2.988 s | 857 MB (rss) | n/a (no oracle) | 0.57x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 371.68 ms / 434.55 ms | 4.072 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 26.21 ms / 26.86 ms | 7.409 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 211.63 ms / 216.22 ms | 5.335 s | 857 MB (rss) | n/a (no oracle) | 0.49x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 429.48 ms / 438.44 ms | 5.157 s | 857 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 49.14 ms / 49.39 ms | 10.265 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | ok | 1.596 s / 1.610 s | 4.548 s | 857 MB (rss) | n/a (no oracle) | 0.41x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 3.939 s / 3.965 s | 7.217 s | 1147 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=20 | `nitrix-jax` | ok | 125.01 ms / 125.19 ms | 12.122 s | 1309.54 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | ok | 625.92 ms / 796.47 ms | 4.885 s | 857 MB (rss) | n/a (no oracle) | 0.14x vs nitrix-jax |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cpu | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 4.509 s / 4.762 s | 8.760 s | 1132 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[96, 96, 96],moving_shape=[80, 96, 112],levels=2,iters=20,space=world,fixed_spacing=[1, 1, 1],moving_spacing=[1.2, 1.0, 0.9] | `nitrix-jax` | ok | 125.42 ms / 127.32 ms | 16.022 s | 1514.57 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

