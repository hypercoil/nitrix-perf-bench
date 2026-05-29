# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 74c2a463e4261e4ded1c4c38d8d6b1febd26235c
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T01:44:10.288576+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| median_filter | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.median_filter` | skipped | — | — | — | — | — |
| median_filter | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 27.72 ms / 28.46 ms | 195.59 ms | 508 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.median_filter` | ok | 7.18 ms / 7.23 ms | 7.42 ms | 447 MB (rss) | n/a (no oracle) | 0.26x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.median_filter` | ok | 61.1 µs / 65.0 µs | 235.31 ms | 0.26 MB (hbm) | n/a (no oracle) | 0.19x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 318.9 µs / 340.0 µs | 538.46 ms | 72.09 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.median_filter` | ok | 7.19 ms / 7.21 ms | 7.40 ms | 0.26 MB (hbm) | n/a (no oracle) | 22.53x vs nitrix-jax |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.median_filter` | skipped | — | — | — | — | — |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 1.78 ms / 1.81 ms | 131.19 ms | 504 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.median_filter` | ok | 428.8 µs / 432.6 µs | 618.8 µs | 447 MB (rss) | n/a (no oracle) | 0.24x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.median_filter` | ok | 61.0 µs / 63.4 µs | 2.794 s | 0.02 MB (hbm) | n/a (no oracle) | 0.56x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 108.0 µs / 109.4 µs | 502.89 ms | 33.87 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.median_filter` | ok | 428.5 µs / 435.4 µs | 615.2 µs | 0.02 MB (hbm) | n/a (no oracle) | 3.97x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

