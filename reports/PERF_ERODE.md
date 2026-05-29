# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:42:01.950650+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| erode | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 742.6 µs / 780.7 µs | 156.79 ms | 505 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.31 ms / 1.36 ms | 1.40 ms | 449 MB (rss) | ✓ 0×tol | 1.76x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 119.3 µs / 123.4 µs | 304.48 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.78x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 153.6 µs / 156.0 µs | 627.84 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.23 ms / 1.26 ms | 1.38 ms | 0.26 MB (hbm) | ✓ 0×tol | 8.03x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 58.1 µs / 59.5 µs | 119.32 ms | 507 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 35.3 µs / 39.9 µs | 93.9 µs | 449 MB (rss) | ✓ 0×tol | 0.61x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 119.0 µs / 123.6 µs | 5.026 s | 0.02 MB (hbm) | ✓ 0×tol | 0.84x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 142.2 µs / 152.3 µs | 834.33 ms | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 58.4 µs / 60.9 µs | 109.3 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.41x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

