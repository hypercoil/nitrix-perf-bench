# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 3ebfa0efd8d768309c2310eb5850e5e4482e759d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T07:17:16.649429+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| intensity_normalize | jax-cpu | n=2048 | `cupy.intensity_normalize` | skipped | — | — | — | — | — |
| intensity_normalize | jax-cpu | n=2048 | `nitrix-jax` | ok | 1.178 s / 1.180 s | 1.508 s | 1151 MB (rss) | ✓ 9.8e-05×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cpu | n=2048 | `numpy.intensity` | ok | 94.99 ms / 98.04 ms | 120.34 ms | 1151 MB (rss) | ✓ 9.8e-05×tol | 0.08x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=2048 | `cupy.intensity_normalize` | ok | 862.6 µs / 870.7 µs | 217.82 ms | 16.78 MB (hbm) | ✓ 9.8e-05×tol | 0.64x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=2048 | `nitrix-jax` | ok | 1.35 ms / 1.36 ms | 469.79 ms | 184.55 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=2048 | `numpy.intensity` | ok | 93.58 ms / 93.92 ms | 109.60 ms | 16.78 MB (hbm) | ✓ 9.8e-05×tol | 69.07x vs nitrix-jax |
| intensity_normalize | jax-cpu | n=4096 | `cupy.intensity_normalize` | skipped | — | — | — | — | — |
| intensity_normalize | jax-cpu | n=4096 | `nitrix-jax` | ok | 5.149 s / 5.157 s | 5.441 s | 1254 MB (rss) | ✓ 0.0001×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cpu | n=4096 | `numpy.intensity` | ok | 402.94 ms / 418.94 ms | 411.07 ms | 1151 MB (rss) | ✓ 0.0001×tol | 0.08x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=4096 | `cupy.intensity_normalize` | ok | 7.90 ms / 7.92 ms | 241.03 ms | 67.11 MB (hbm) | ✓ 0.0001×tol | 1.18x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=4096 | `nitrix-jax` | ok | 6.70 ms / 6.72 ms | 400.61 ms | 539.51 MB (hbm) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=4096 | `numpy.intensity` | ok | 398.42 ms / 399.75 ms | 400.62 ms | 67.11 MB (hbm) | ✓ 0.0001×tol | 59.44x vs nitrix-jax |
| intensity_normalize | jax-cpu | n=512 | `cupy.intensity_normalize` | skipped | — | — | — | — | — |
| intensity_normalize | jax-cpu | n=512 | `nitrix-jax` | ok | 58.94 ms / 59.50 ms | 267.81 ms | 1151 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cpu | n=512 | `numpy.intensity` | ok | 5.06 ms / 5.10 ms | 5.13 ms | 1151 MB (rss) | ✓ 0.00011×tol | 0.09x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=512 | `cupy.intensity_normalize` | ok | 417.7 µs / 425.0 µs | 1.412 s | 1.05 MB (hbm) | ✓ 0.00011×tol | 2.21x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=512 | `nitrix-jax` | ok | 189.4 µs / 197.7 µs | 501.96 ms | 34.60 MB (hbm) | ✓ 0.00015×tol | 1.00x vs nitrix-jax |
| intensity_normalize | jax-cuda12 | n=512 | `numpy.intensity` | ok | 5.55 ms / 5.67 ms | 5.78 ms | 1.05 MB (hbm) | ✓ 0.00011×tol | 29.28x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

