# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: fa0a230c55f2a768bb14670d07c7d23c6f102590
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T03:11:45.272754+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `nitrix-jax` | ok | 10.79 ms / 19.70 ms | 517.70 ms | 1209 MB (rss) | ✓ 0.33×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 180.21 ms / 215.32 ms | 282.19 ms | 1209 MB (rss) | ✓ 0.00092×tol | 16.70x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `cupy.lstsq_detrend` | ok | 4.61 ms / 4.63 ms | 367.96 ms | 16.78 MB (hbm) | ✓ 0.00092×tol | 12.12x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `nitrix-jax` | ok | 380.1 µs / 386.3 µs | 1.051 s | 121.90 MB (hbm) | ✓ 0.024×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 164.96 ms / 165.86 ms | 161.69 ms | 16.78 MB (hbm) | ✓ 0.00092×tol | 433.93x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `nitrix-jax` | ok | 711.7 µs / 739.3 µs | 246.63 ms | 1209 MB (rss) | ✓ 0.19×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `numpy.lstsq_detrend` | ok | 11.50 ms / 12.43 ms | 12.91 ms | 1209 MB (rss) | ✓ 0.0032×tol | 16.17x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `cupy.lstsq_detrend` | ok | 1.69 ms / 1.70 ms | 1.719 s | 2.10 MB (hbm) | ✓ 0.0032×tol | 11.10x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `nitrix-jax` | ok | 151.9 µs / 164.7 µs | 1.047 s | 77.59 MB (hbm) | ✓ 0.025×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `numpy.lstsq_detrend` | ok | 11.20 ms / 28.10 ms | 11.98 ms | 2.10 MB (hbm) | ✓ 0.0032×tol | 73.71x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `nitrix-jax` | ok | 92.62 ms / 131.30 ms | 414.49 ms | 1209 MB (rss) | ✓ 0.31×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 754.97 ms / 813.87 ms | 938.12 ms | 1209 MB (rss) | ✓ 0.00092×tol | 8.15x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `cupy.lstsq_detrend` | ok | 14.26 ms / 14.33 ms | 348.25 ms | 67.11 MB (hbm) | ✓ 0.00092×tol | 6.84x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `nitrix-jax` | ok | 2.09 ms / 2.10 ms | 940.15 ms | 268.83 MB (hbm) | ✓ 0.03×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 728.96 ms / 732.83 ms | 1.070 s | 67.11 MB (hbm) | ✓ 0.00092×tol | 349.52x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

