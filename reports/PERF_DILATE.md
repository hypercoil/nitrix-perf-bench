# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:42:27.898424+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 827.8 µs / 891.0 µs | 173.35 ms | 507 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.22 ms / 1.23 ms | 1.27 ms | 449 MB (rss) | ✓ 0×tol | 1.47x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 126.1 µs / 131.1 µs | 229.38 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.83x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 152.8 µs / 167.9 µs | 621.99 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.21 ms / 1.22 ms | 1.24 ms | 0.26 MB (hbm) | ✓ 0×tol | 7.93x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 84.3 µs / 85.2 µs | 120.81 ms | 506 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 38.9 µs / 40.9 µs | 104.4 µs | 449 MB (rss) | ✓ 0×tol | 0.46x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 122.6 µs / 127.7 µs | 4.846 s | 0.02 MB (hbm) | ✓ 0×tol | 0.84x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 145.8 µs / 148.0 µs | 550.32 ms | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 71.3 µs / 82.1 µs | 125.6 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.49x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

