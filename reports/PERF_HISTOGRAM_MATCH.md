# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: be1403238ab822dfd4d8ce256fe79dbb69bacbbf | bench: 8d7c2e44607e6159bfebb98fb9b3dfedaac147ed
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T21:39:16.989565+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| histogram_match | jax-cpu | n=32 | `nitrix-jax` | ok | 799.0 µs / 860.3 µs | 394.15 ms | 719 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cpu | n=32 | `simpleitk.HistogramMatching` | ok | 6.12 ms / 6.64 ms | 72.89 ms | 719 MB (rss) | n/a (no oracle) | 7.65x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=32 | `nitrix-jax` | ok | 407.4 µs / 416.0 µs | 705.35 ms | 33.94 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=32 | `simpleitk.HistogramMatching` | ok | 6.38 ms / 8.17 ms | 71.55 ms | 0.39 MB (hbm) | n/a (no oracle) | 15.65x vs nitrix-jax |
| histogram_match | jax-cpu | n=64 | `nitrix-jax` | ok | 3.59 ms / 4.15 ms | 463.52 ms | 719 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cpu | n=64 | `simpleitk.HistogramMatching` | ok | 38.99 ms / 43.38 ms | 103.04 ms | 719 MB (rss) | n/a (no oracle) | 10.87x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=64 | `nitrix-jax` | ok | 893.2 µs / 895.5 µs | 1.093 s | 36.10 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=64 | `simpleitk.HistogramMatching` | ok | 38.63 ms / 44.64 ms | 106.55 ms | 2.54 MB (hbm) | n/a (no oracle) | 43.25x vs nitrix-jax |
| histogram_match | jax-cpu | n=96 | `nitrix-jax` | ok | 11.05 ms / 12.61 ms | 502.45 ms | 748 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cpu | n=96 | `simpleitk.HistogramMatching` | ok | 121.17 ms / 125.40 ms | 191.75 ms | 719 MB (rss) | n/a (no oracle) | 10.96x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=96 | `nitrix-jax` | ok | 2.11 ms / 2.13 ms | 1.021 s | 46.14 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| histogram_match | jax-cuda12 | n=96 | `simpleitk.HistogramMatching` | ok | 117.37 ms / 118.49 ms | 178.81 ms | 12.58 MB (hbm) | n/a (no oracle) | 55.54x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

