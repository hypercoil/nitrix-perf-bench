# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 7a601d42ba56b1a3c7acd90a5fd9fac64d78aced | bench: b2dfe3ad993496df417e06bde549e2183a2d1985
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T01:05:35.838107+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| bilateral_gaussian | jax-cpu | shape=[128, 128],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 11.08 ms / 13.17 ms | 408.41 ms | 1025 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cpu | shape=[128, 128],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 7.25 ms / 10.81 ms | 64.87 ms | 1025 MB (rss) | n/a (no oracle) | 0.65x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[128, 128],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 673.5 µs / 677.6 µs | 692.74 ms | 124.72 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[128, 128],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 10.51 ms / 10.65 ms | 75.42 ms | 17.04 MB (hbm) | n/a (no oracle) | 15.61x vs nitrix-jax |
| bilateral_gaussian | jax-cpu | shape=[256, 256],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 29.38 ms / 33.96 ms | 1.411 s | 1083 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cpu | shape=[256, 256],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 30.92 ms / 35.57 ms | 104.51 ms | 1025 MB (rss) | n/a (no oracle) | 1.05x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[256, 256],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 1.66 ms / 1.72 ms | 1.867 s | 215.48 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[256, 256],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 22.36 ms / 23.75 ms | 75.01 ms | 68.16 MB (hbm) | n/a (no oracle) | 13.48x vs nitrix-jax |
| bilateral_gaussian | jax-cpu | shape=[64, 64],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 1.57 ms / 1.60 ms | 170.48 ms | 1025 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cpu | shape=[64, 64],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 3.53 ms / 3.93 ms | 57.98 ms | 1025 MB (rss) | n/a (no oracle) | 2.25x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[64, 64],sigma_d=2.0,sigma_r=0.2 | `nitrix-jax` | ok | 612.7 µs / 702.7 µs | 634.86 ms | 94.14 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bilateral_gaussian | jax-cuda12 | shape=[64, 64],sigma_d=2.0,sigma_r=0.2 | `simpleitk.Bilateral` | ok | 3.85 ms / 4.44 ms | 57.50 ms | 4.14 MB (hbm) | n/a (no oracle) | 6.28x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

