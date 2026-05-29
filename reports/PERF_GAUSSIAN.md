# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:41:21.197895+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | skipped | — | — | — | — | — |
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `nitrix-jax` | ok | 2.98 ms / 3.77 ms | 163.45 ms | 530 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 704.0 µs / 709.4 µs | 752.2 µs | 454 MB (rss) | ✓ 0.00021×tol | 0.24x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[256, 256],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | ok | 254.8 µs / 258.4 µs | 263.55 ms | 0.26 MB (hbm) | ✓ 0.00078×tol | 2.03x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[256, 256],sigma=1.5 | `nitrix-jax` | ok | 125.8 µs / 127.3 µs | 564.22 ms | 69.21 MB (hbm) | ✓ 0.00072×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[256, 256],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 1.05 ms / 1.06 ms | 1.13 ms | 0.26 MB (hbm) | ✓ 0.00021×tol | 8.33x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | skipped | — | — | — | — | — |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `nitrix-jax` | ok | 16.24 ms / 18.27 ms | 162.86 ms | 597 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 4.61 ms / 4.66 ms | 4.73 ms | 454 MB (rss) | ✓ 0.00016×tol | 0.28x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64, 64],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | ok | 361.1 µs / 370.6 µs | 7.125 s | 1.05 MB (hbm) | ✓ 0.00053×tol | 2.20x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64, 64],sigma=1.5 | `nitrix-jax` | ok | 164.4 µs / 166.3 µs | 699.67 ms | 615.65 MB (hbm) | ✓ 0.00055×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 4.62 ms / 4.67 ms | 4.79 ms | 1.05 MB (hbm) | ✓ 0.00016×tol | 28.08x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | skipped | — | — | — | — | — |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `nitrix-jax` | ok | 364.2 µs / 367.4 µs | 122.70 ms | 506 MB (rss) | ✓ 0.00047×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 61.0 µs / 63.4 µs | 105.9 µs | 454 MB (rss) | ✓ 0.00019×tol | 0.17x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64],sigma=1.5 | `cupyx.scipy.ndimage.gaussian_filter` | ok | 256.9 µs / 264.6 µs | 5.692 s | 0.02 MB (hbm) | ✓ 0.00052×tol | 2.15x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64],sigma=1.5 | `nitrix-jax` | ok | 119.7 µs / 125.0 µs | 1.369 s | 75.81 MB (hbm) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cuda12 | shape=[64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 98.6 µs / 102.5 µs | 145.8 µs | 0.02 MB (hbm) | ✓ 0.00019×tol | 0.82x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

