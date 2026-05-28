# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: f9cc83fb07be9f33fba7916ff60a91d3d5136274
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-28T23:14:46.525864+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 843.2 µs / 909.6 µs | 150.85 ms | 514 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.22 ms / 1.22 ms | 1.28 ms | 449 MB (rss) | ✓ 0×tol | 1.44x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 151.9 µs / 153.6 µs | 552.45 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.20 ms / 1.22 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 7.93x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 64.1 µs / 66.1 µs | 123.08 ms | 507 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 55.8 µs / 58.9 µs | 115.4 µs | 449 MB (rss) | ✓ 0×tol | 0.87x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 143.1 µs / 151.2 µs | 543.44 ms | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 38.7 µs / 40.2 µs | 117.6 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.27x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

