# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: 44def4b7ce5c1f37844a65f1545ddc4ba9281c5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T22:38:09.061263+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform | jax-cpu | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 25.10 ms / 29.01 ms | 228.75 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 917.6 µs / 979.6 µs | 1.43 ms | 712 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.92 ms / 1.95 ms | 61.37 ms | 712 MB (rss) | ✓ 3.2e-08×tol | 0.08x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 201.7 µs / 208.0 µs | 127.89 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 4.99 ms / 5.00 ms | 562.98 ms | 68.35 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_edt` | ok | 981.7 µs / 1.37 ms | 1.52 ms | 0.07 MB (hbm) | ✓ 0×tol | 0.20x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[128, 128] | `simpleitk.DanielssonDistanceMap` | ok | 1.95 ms / 2.11 ms | 59.24 ms | 0.07 MB (hbm) | ✓ 3.2e-08×tol | 0.39x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `nitrix-jax` | ok | 85.78 ms / 87.54 ms | 314.87 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.51 ms / 4.41 ms | 4.91 ms | 712 MB (rss) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 10.14 ms / 10.48 ms | 67.56 ms | 712 MB (rss) | ✓ 2.4e-08×tol | 0.12x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 146.9 µs / 168.5 µs | 155.57 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `nitrix-jax` | ok | 3.91 ms / 3.92 ms | 609.23 ms | 74.32 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 3.37 ms / 4.52 ms | 5.12 ms | 0.13 MB (hbm) | ✓ 0×tol | 0.86x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 10.21 ms / 10.36 ms | 70.34 ms | 0.13 MB (hbm) | ✓ 2.4e-08×tol | 2.61x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[32, 32] | `nitrix-jax` | ok | 426.8 µs / 446.8 µs | 173.86 ms | 712 MB (rss) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 58.3 µs / 70.2 µs | 103.2 µs | 712 MB (rss) | ✓ 0×tol | 0.14x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 565.1 µs / 592.0 µs | 56.73 ms | 712 MB (rss) | ✓ 2.4e-08×tol | 1.32x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 184.3 µs / 191.3 µs | 164.52 ms | 0.00 MB (hbm) | ✓ 0×tol | 0.11x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `nitrix-jax` | ok | 1.62 ms / 1.66 ms | 547.11 ms | 33.63 MB (hbm) | ✓ 0.41×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `scipy.ndimage.distance_transform_edt` | ok | 81.0 µs / 90.7 µs | 129.5 µs | 0.00 MB (hbm) | ✓ 0×tol | 0.05x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[32, 32] | `simpleitk.DanielssonDistanceMap` | ok | 553.3 µs / 566.8 µs | 57.50 ms | 0.00 MB (hbm) | ✓ 2.4e-08×tol | 0.34x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | skipped | — | — | — | — | — |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 1.182 s / 1.530 s | 1.743 s | 712 MB (rss) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 31.01 ms / 33.50 ms | 42.59 ms | 712 MB (rss) | ✓ 0×tol | 0.03x vs nitrix-jax |
| distance_transform | jax-cpu | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 80.66 ms / 81.50 ms | 140.27 ms | 712 MB (rss) | ✓ 3.1e-08×tol | 0.07x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `cupyx.scipy.ndimage.distance_transform_edt` | ok | 215.6 µs / 223.1 µs | 153.59 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.01x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 23.39 ms / 23.48 ms | 738.78 ms | 135.27 MB (hbm) | ✓ 0.72×tol | 1.00x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_edt` | ok | 29.95 ms / 30.89 ms | 36.92 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.28x vs nitrix-jax |
| distance_transform | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.DanielssonDistanceMap` | ok | 80.94 ms / 81.78 ms | 141.19 ms | 1.05 MB (hbm) | ✓ 3.1e-08×tol | 3.46x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

