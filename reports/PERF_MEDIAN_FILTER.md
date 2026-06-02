# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: 44def4b7ce5c1f37844a65f1545ddc4ba9281c5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T22:37:25.196381+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| median_filter | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.median_filter` | skipped | — | — | — | — | — |
| median_filter | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 27.85 ms / 36.02 ms | 200.13 ms | 696 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.median_filter` | ok | 7.98 ms / 8.14 ms | 7.90 ms | 696 MB (rss) | n/a (no oracle) | 0.29x vs nitrix-jax |
| median_filter | jax-cpu | shape=[256, 256],size=3 | `simpleitk.Median` | ok | 2.39 ms / 2.41 ms | 58.98 ms | 696 MB (rss) | n/a (no oracle) | 0.09x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.median_filter` | ok | 62.9 µs / 65.0 µs | 150.15 ms | 0.26 MB (hbm) | n/a (no oracle) | 0.20x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 312.3 µs / 338.8 µs | 577.03 ms | 72.09 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.median_filter` | ok | 7.19 ms / 7.23 ms | 7.38 ms | 0.26 MB (hbm) | n/a (no oracle) | 23.02x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[256, 256],size=3 | `simpleitk.Median` | ok | 2.41 ms / 2.58 ms | 59.41 ms | 0.26 MB (hbm) | n/a (no oracle) | 7.71x vs nitrix-jax |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.median_filter` | skipped | — | — | — | — | — |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 1.76 ms / 1.77 ms | 143.39 ms | 696 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.median_filter` | ok | 472.8 µs / 566.7 µs | 711.8 µs | 696 MB (rss) | n/a (no oracle) | 0.27x vs nitrix-jax |
| median_filter | jax-cpu | shape=[64, 64],size=3 | `simpleitk.Median` | ok | 502.9 µs / 532.5 µs | 64.24 ms | 696 MB (rss) | n/a (no oracle) | 0.29x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.median_filter` | ok | 59.1 µs / 62.7 µs | 129.36 ms | 0.02 MB (hbm) | n/a (no oracle) | 0.42x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 141.2 µs / 142.6 µs | 491.66 ms | 33.87 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.median_filter` | ok | 551.8 µs / 557.4 µs | 741.5 µs | 0.02 MB (hbm) | n/a (no oracle) | 3.91x vs nitrix-jax |
| median_filter | jax-cuda12 | shape=[64, 64],size=3 | `simpleitk.Median` | ok | 500.2 µs / 509.7 µs | 57.70 ms | 0.02 MB (hbm) | n/a (no oracle) | 3.54x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

