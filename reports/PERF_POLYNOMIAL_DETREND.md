# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 8259a204821af34c01e8092f9cf37275f388413d | bench: 5e5d31ee4e691c0e081dc112fbadc07d6cc03be7
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T23:44:47.122849+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `nilearn.signal_clean` | ok | 518.90 ms / 526.87 ms | 677.85 ms | 1406 MB (rss) | ✓ 0.00092×tol | 32.15x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `nitrix-jax` | ok | 16.14 ms / 34.43 ms | 444.55 ms | 1406 MB (rss) | ✓ 0.33×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=1024,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 178.76 ms / 202.19 ms | 212.60 ms | 1406 MB (rss) | ✓ 0.00092×tol | 11.08x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `cupy.lstsq_detrend` | ok | 4.32 ms / 4.34 ms | 311.55 ms | 16.78 MB (hbm) | ✓ 0.00092×tol | 11.33x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `nilearn.signal_clean` | ok | 557.81 ms / 622.15 ms | 991.95 ms | 16.78 MB (hbm) | ✓ 0.00092×tol | 1463.54x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `nitrix-jax` | ok | 381.1 µs / 386.2 µs | 995.36 ms | 121.90 MB (hbm) | ✓ 0.024×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=1024,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 162.73 ms / 163.57 ms | 162.57 ms | 16.78 MB (hbm) | ✓ 0.00092×tol | 426.96x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `nilearn.signal_clean` | ok | 54.98 ms / 55.27 ms | 196.83 ms | 1406 MB (rss) | ✓ 0.0032×tol | 79.05x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `nitrix-jax` | ok | 695.5 µs / 874.3 µs | 221.83 ms | 1406 MB (rss) | ✓ 0.19×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=256,obs=2048,degree=3 | `numpy.lstsq_detrend` | ok | 11.88 ms / 15.77 ms | 10.80 ms | 1406 MB (rss) | ✓ 0.0032×tol | 17.07x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `cupy.lstsq_detrend` | ok | 1.68 ms / 1.71 ms | 561.12 ms | 2.10 MB (hbm) | ✓ 0.0032×tol | 10.65x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `nilearn.signal_clean` | ok | 55.46 ms / 76.16 ms | 194.11 ms | 2.10 MB (hbm) | ✓ 0.0032×tol | 351.80x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `nitrix-jax` | ok | 157.6 µs / 167.7 µs | 657.81 ms | 77.59 MB (hbm) | ✓ 0.025×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=256,obs=2048,degree=3 | `numpy.lstsq_detrend` | ok | 10.61 ms / 13.44 ms | 42.72 ms | 2.10 MB (hbm) | ✓ 0.0032×tol | 67.32x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `cupy.lstsq_detrend` | skipped | — | — | — | — | — |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `nilearn.signal_clean` | ok | 2.079 s / 2.487 s | 2.410 s | 1406 MB (rss) | ✓ 0.00092×tol | 14.32x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `nitrix-jax` | ok | 145.19 ms / 176.45 ms | 508.87 ms | 1406 MB (rss) | ✓ 0.31×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cpu | c=4096,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 713.08 ms / 716.14 ms | 709.28 ms | 1406 MB (rss) | ✓ 0.00092×tol | 4.91x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `cupy.lstsq_detrend` | ok | 14.28 ms / 14.34 ms | 332.33 ms | 67.11 MB (hbm) | ✓ 0.00092×tol | 6.84x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `nilearn.signal_clean` | ok | 1.768 s / 1.775 s | 1.953 s | 67.11 MB (hbm) | ✓ 0.00092×tol | 847.60x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `nitrix-jax` | ok | 2.09 ms / 2.10 ms | 1.068 s | 268.83 MB (hbm) | ✓ 0.03×tol | 1.00x vs nitrix-jax |
| polynomial_detrend | jax-cuda12 | c=4096,obs=4096,degree=3 | `numpy.lstsq_detrend` | ok | 907.56 ms / 1.075 s | 1.411 s | 67.11 MB (hbm) | ✓ 0.00092×tol | 435.01x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

