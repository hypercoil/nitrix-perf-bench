# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 2dd4fc7b7820e676dd2c5a3b51b4ef8bd7bd8d1f | bench: 1ea305473454e971b7e7c431c961d9d04b7e5e13
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-06T06:58:23.460898+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sosfiltfilt | jax-cpu | channels=256,obs=32768,order=4 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax` | ok | 194.29 ms / 206.30 ms | 536.69 ms | 1070 MB (rss) | ✓ 0.015×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax-fft` | ok | 259.72 ms / 285.63 ms | 556.70 ms | 1070 MB (rss) | ✓ 0.0081×tol | 1.34x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax-scan` | ok | 181.30 ms / 201.59 ms | 519.62 ms | 1070 MB (rss) | ✓ 0.015×tol | 0.93x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=32768,order=4 | `scipy.signal.sosfiltfilt` | ok | 159.73 ms / 161.85 ms | 158.85 ms | 1070 MB (rss) | ✓ 0.023×tol | 0.82x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `cupyx.scipy.signal.sosfiltfilt` | ok | 12.62 ms / 13.30 ms | 2.920 s | 33.55 MB (hbm) | ✓ 0.2×tol | 1.07x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax` | ok | 11.83 ms / 11.97 ms | 403.21 ms | 469.77 MB (hbm) | ✓ 0.013×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax-fft` | ok | 11.83 ms / 11.93 ms | 283.78 ms | 469.77 MB (hbm) | ✓ 0.013×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax-scan` | ok | 2.401 s / 2.625 s | 3.850 s | 403.16 MB (hbm) | ✓ 0.023×tol | 202.96x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `scipy.signal.sosfiltfilt` | ok | 160.64 ms / 162.23 ms | 159.76 ms | 33.55 MB (hbm) | ✓ 0.023×tol | 13.58x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=8192,order=4 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax` | ok | 48.60 ms / 52.84 ms | 520.80 ms | 1070 MB (rss) | ✓ 0.02×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax-fft` | ok | 47.66 ms / 63.87 ms | 244.58 ms | 1070 MB (rss) | ✓ 0.0071×tol | 0.98x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax-scan` | ok | 38.10 ms / 40.36 ms | 347.75 ms | 1070 MB (rss) | ✓ 0.02×tol | 0.78x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=256,obs=8192,order=4 | `scipy.signal.sosfiltfilt` | ok | 28.71 ms / 29.79 ms | 32.82 ms | 1070 MB (rss) | ✓ 0.021×tol | 0.59x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `cupyx.scipy.signal.sosfiltfilt` | ok | 6.37 ms / 6.43 ms | 2.791 s | 8.39 MB (hbm) | ✓ 0.19×tol | 7.37x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax` | ok | 864.9 µs / 885.3 µs | 343.66 ms | 100.67 MB (hbm) | ✓ 0.015×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax-fft` | ok | 846.0 µs / 858.8 µs | 254.30 ms | 100.67 MB (hbm) | ✓ 0.015×tol | 0.98x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax-scan` | ok | 705.56 ms / 755.95 ms | 1.474 s | 159.38 MB (hbm) | ✓ 0.021×tol | 815.73x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `scipy.signal.sosfiltfilt` | ok | 28.59 ms / 29.35 ms | 33.17 ms | 8.39 MB (hbm) | ✓ 0.021×tol | 33.06x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=4096,order=4 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax` | ok | 5.25 ms / 5.96 ms | 425.61 ms | 1070 MB (rss) | ✓ 0.018×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax-fft` | ok | 5.45 ms / 6.30 ms | 251.14 ms | 1070 MB (rss) | ✓ 0.0055×tol | 1.04x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax-scan` | ok | 5.06 ms / 5.85 ms | 400.11 ms | 1070 MB (rss) | ✓ 0.018×tol | 0.96x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=4096,order=4 | `scipy.signal.sosfiltfilt` | ok | 3.95 ms / 3.97 ms | 4.49 ms | 1070 MB (rss) | ✓ 0.017×tol | 0.75x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `cupyx.scipy.signal.sosfiltfilt` | ok | 5.49 ms / 5.96 ms | 3.068 s | 1.05 MB (hbm) | ✓ 0.17×tol | 27.23x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax` | ok | 201.7 µs / 210.4 µs | 360.73 ms | 12.58 MB (hbm) | ✓ 0.011×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax-fft` | ok | 210.6 µs / 231.8 µs | 353.33 ms | 12.58 MB (hbm) | ✓ 0.011×tol | 1.04x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax-scan` | ok | 320.70 ms / 369.77 ms | 1.065 s | 36.70 MB (hbm) | ✓ 0.017×tol | 1590.16x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `scipy.signal.sosfiltfilt` | ok | 3.92 ms / 4.10 ms | 4.19 ms | 1.05 MB (hbm) | ✓ 0.017×tol | 19.43x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=65536,order=8 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax` | ok | 204.39 ms / 237.79 ms | 758.39 ms | 1070 MB (rss) | ✓ 0.024×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax-fft` | ok | 122.02 ms / 147.11 ms | 574.47 ms | 1070 MB (rss) | ✓ 0.0088×tol | 0.60x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax-scan` | ok | 201.16 ms / 236.31 ms | 817.76 ms | 1070 MB (rss) | ✓ 0.024×tol | 0.98x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=64,obs=65536,order=8 | `scipy.signal.sosfiltfilt` | ok | 93.40 ms / 94.69 ms | 120.14 ms | 1070 MB (rss) | ✓ 0.035×tol | 0.46x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `cupyx.scipy.signal.sosfiltfilt` | ok | 15.55 ms / 15.91 ms | 2.938 s | 16.78 MB (hbm) | ✓ 0.25×tol | 4.16x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax` | ok | 3.74 ms / 3.78 ms | 356.18 ms | 234.88 MB (hbm) | ✓ 0.015×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax-fft` | ok | 3.69 ms / 3.72 ms | 352.34 ms | 234.88 MB (hbm) | ✓ 0.015×tol | 0.99x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax-scan` | ok | 9.511 s / 9.570 s | 10.171 s | 336.00 MB (hbm) | ✓ 0.035×tol | 2546.18x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `scipy.signal.sosfiltfilt` | ok | 92.89 ms / 93.48 ms | 118.52 ms | 16.78 MB (hbm) | ✓ 0.035×tol | 24.87x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

