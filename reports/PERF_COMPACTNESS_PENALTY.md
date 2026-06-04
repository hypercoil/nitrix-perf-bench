# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 8171f3faa23e2dd8c88e3ef9a4602a7b487c0b85
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T05:40:02.443543+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| compactness_penalty | jax-cpu | p=1024 | `cupy.compactness_penalty` | skipped | — | — | — | — | — |
| compactness_penalty | jax-cpu | p=1024 | `nitrix-jax` | ok | 99.0 µs / 103.1 µs | 109.55 ms | 774 MB (rss) | ✓ 0.00019×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cpu | p=1024 | `numpy.compactness` | ok | 2.16 ms / 2.16 ms | 2.24 ms | 774 MB (rss) | ✓ 5.9e-05×tol | 21.81x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=1024 | `cupy.compactness_penalty` | ok | 276.4 µs / 281.8 µs | 460.66 ms | 0.27 MB (hbm) | ✓ 9.6e-05×tol | 2.37x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=1024 | `nitrix-jax` | ok | 116.4 µs / 124.4 µs | 517.27 ms | 100.94 MB (hbm) | ✓ 7.6e-05×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=1024 | `numpy.compactness` | ok | 2.34 ms / 2.35 ms | 3.04 ms | 0.27 MB (hbm) | ✓ 5.9e-05×tol | 20.14x vs nitrix-jax |
| compactness_penalty | jax-cpu | p=16384 | `cupy.compactness_penalty` | skipped | — | — | — | — | — |
| compactness_penalty | jax-cpu | p=16384 | `nitrix-jax` | ok | 930.5 µs / 1.98 ms | 183.39 ms | 774 MB (rss) | ✓ 7.6e-05×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cpu | p=16384 | `numpy.compactness` | ok | 31.41 ms / 41.47 ms | 35.44 ms | 774 MB (rss) | ✓ 7e-05×tol | 33.76x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=16384 | `cupy.compactness_penalty` | ok | 1.91 ms / 1.93 ms | 237.06 ms | 4.39 MB (hbm) | ✓ 9.1e-05×tol | 14.35x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=16384 | `nitrix-jax` | ok | 133.4 µs / 140.2 µs | 681.70 ms | 75.74 MB (hbm) | ✓ 7.4e-05×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=16384 | `numpy.compactness` | ok | 29.84 ms / 30.00 ms | 30.07 ms | 4.39 MB (hbm) | ✓ 7e-05×tol | 223.76x vs nitrix-jax |
| compactness_penalty | jax-cpu | p=4096 | `cupy.compactness_penalty` | skipped | — | — | — | — | — |
| compactness_penalty | jax-cpu | p=4096 | `nitrix-jax` | ok | 250.4 µs / 262.9 µs | 169.71 ms | 774 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cpu | p=4096 | `numpy.compactness` | ok | 7.59 ms / 7.82 ms | 7.65 ms | 774 MB (rss) | ✓ 6.5e-05×tol | 30.31x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=4096 | `cupy.compactness_penalty` | ok | 592.0 µs / 599.4 µs | 214.10 ms | 1.10 MB (hbm) | ✓ 0.00012×tol | 5.09x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=4096 | `nitrix-jax` | ok | 116.4 µs / 120.3 µs | 643.35 ms | 84.99 MB (hbm) | ✓ 8.4e-05×tol | 1.00x vs nitrix-jax |
| compactness_penalty | jax-cuda12 | p=4096 | `numpy.compactness` | ok | 7.39 ms / 7.54 ms | 8.25 ms | 1.10 MB (hbm) | ✓ 6.5e-05×tol | 63.52x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

