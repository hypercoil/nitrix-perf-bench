# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 2dd4fc7b7820e676dd2c5a3b51b4ef8bd7bd8d1f | bench: 1ea305473454e971b7e7c431c961d9d04b7e5e13
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-06T06:50:37.347857+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax` | ok | 184.94 ms / 223.83 ms | 725.05 ms | 1022 MB (rss) | ✓ 0.0011×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax-assoc` | ok | 902.10 ms / 1.047 s | 4.015 s | 1903 MB (rss) | ✓ 0.026×tol | 4.88x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax-fft` | ok | 122.44 ms / 150.38 ms | 301.75 ms | 946 MB (rss) | ✓ 0.0062×tol | 0.66x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `nitrix-jax-scan` | ok | 157.18 ms / 173.61 ms | 574.36 ms | 1025 MB (rss) | ✓ 0.0011×tol | 0.85x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=32768,order=4 | `scipy.signal.sosfilt` | ok | 72.60 ms / 93.88 ms | 68.76 ms | 946 MB (rss) | ✓ 0.023×tol | 0.39x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `cupyx.scipy.signal.sosfilt` | ok | 3.84 ms / 4.31 ms | 3.865 s | 33.55 MB (hbm) | ✓ 0.18×tol | 0.62x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax` | ok | 6.14 ms / 6.21 ms | 315.38 ms | 469.77 MB (hbm) | ✓ 0.013×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax-assoc` | ok | 31.55 ms / 31.67 ms | 9.184 s | 385.90 MB (hbm) | ✓ 0.021×tol | 5.14x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax-fft` | ok | 6.09 ms / 6.10 ms | 199.14 ms | 469.77 MB (hbm) | ✓ 0.013×tol | 0.99x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `nitrix-jax-scan` | ok | 1.607 s / 1.928 s | 2.905 s | 436.21 MB (hbm) | ✓ 0.0011×tol | 261.52x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=32768,order=4 | `scipy.signal.sosfilt` | ok | 74.38 ms / 77.71 ms | 75.16 ms | 33.55 MB (hbm) | ✓ 0.023×tol | 12.11x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax` | ok | 37.26 ms / 44.15 ms | 400.35 ms | 946 MB (rss) | ✓ 0.0011×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax-assoc` | ok | 232.17 ms / 256.93 ms | 2.847 s | 1048 MB (rss) | ✓ 0.023×tol | 6.23x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax-fft` | ok | 41.56 ms / 45.07 ms | 163.63 ms | 946 MB (rss) | ✓ 0.006×tol | 1.12x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `nitrix-jax-scan` | ok | 40.24 ms / 48.41 ms | 375.00 ms | 946 MB (rss) | ✓ 0.0011×tol | 1.08x vs nitrix-jax |
| sosfilt | jax-cpu | channels=256,obs=8192,order=4 | `scipy.signal.sosfilt` | ok | 13.52 ms / 17.46 ms | 18.74 ms | 946 MB (rss) | ✓ 0.019×tol | 0.36x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `cupyx.scipy.signal.sosfilt` | ok | 1.01 ms / 1.07 ms | 4.018 s | 8.39 MB (hbm) | ✓ 0.17×tol | 2.16x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax` | ok | 468.5 µs / 475.2 µs | 321.81 ms | 100.67 MB (hbm) | ✓ 0.01×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax-assoc` | ok | 4.01 ms / 4.06 ms | 7.932 s | 318.78 MB (hbm) | ✓ 0.021×tol | 8.56x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax-fft` | ok | 486.9 µs / 495.7 µs | 320.73 ms | 100.67 MB (hbm) | ✓ 0.01×tol | 1.04x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `nitrix-jax-scan` | ok | 353.91 ms / 439.01 ms | 961.19 ms | 109.05 MB (hbm) | ✓ 0.0011×tol | 755.46x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=256,obs=8192,order=4 | `scipy.signal.sosfilt` | ok | 13.78 ms / 16.60 ms | 17.52 ms | 8.39 MB (hbm) | ✓ 0.019×tol | 29.41x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax` | ok | 3.18 ms / 4.97 ms | 291.79 ms | 946 MB (rss) | ✓ 0.00077×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax-assoc` | ok | 23.44 ms / 26.19 ms | 2.571 s | 946 MB (rss) | ✓ 0.022×tol | 7.37x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax-fft` | ok | 2.22 ms / 2.40 ms | 126.81 ms | 946 MB (rss) | ✓ 0.004×tol | 0.70x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `nitrix-jax-scan` | ok | 2.81 ms / 3.92 ms | 441.33 ms | 946 MB (rss) | ✓ 0.00077×tol | 0.89x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=4096,order=4 | `scipy.signal.sosfilt` | ok | 1.68 ms / 1.71 ms | 2.33 ms | 946 MB (rss) | ✓ 0.016×tol | 0.53x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `cupyx.scipy.signal.sosfilt` | ok | 308.9 µs / 314.1 µs | 4.850 s | 1.05 MB (hbm) | ✓ 0.12×tol | 1.82x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax` | ok | 170.2 µs / 191.2 µs | 244.63 ms | 12.58 MB (hbm) | ✓ 0.0087×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax-assoc` | ok | 734.0 µs / 739.8 µs | 4.835 s | 287.31 MB (hbm) | ✓ 0.018×tol | 4.31x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax-fft` | ok | 158.3 µs / 293.2 µs | 229.73 ms | 12.58 MB (hbm) | ✓ 0.0087×tol | 0.93x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `nitrix-jax-scan` | ok | 157.32 ms / 209.97 ms | 746.83 ms | 38.80 MB (hbm) | ✓ 0.00077×tol | 924.47x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=4096,order=4 | `scipy.signal.sosfilt` | ok | 2.20 ms / 2.22 ms | 2.28 ms | 1.05 MB (hbm) | ✓ 0.016×tol | 12.91x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax` | ok | 202.65 ms / 245.71 ms | 629.25 ms | 950 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax-assoc` | ok | 834.42 ms / 937.96 ms | 5.519 s | 1890 MB (rss) | ✓ 0.04×tol | 4.12x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax-fft` | ok | 63.85 ms / 97.31 ms | 376.42 ms | 946 MB (rss) | ✓ 0.0061×tol | 0.32x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `nitrix-jax-scan` | ok | 240.04 ms / 291.04 ms | 889.65 ms | 951 MB (rss) | ✓ 0.001×tol | 1.18x vs nitrix-jax |
| sosfilt | jax-cpu | channels=64,obs=65536,order=8 | `scipy.signal.sosfilt` | ok | 48.27 ms / 62.31 ms | 64.15 ms | 946 MB (rss) | ✓ 0.032×tol | 0.24x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `cupyx.scipy.signal.sosfilt` | ok | 3.58 ms / 3.60 ms | 3.367 s | 16.78 MB (hbm) | ✓ 0.23×tol | 1.85x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax` | ok | 1.94 ms / 1.96 ms | 311.61 ms | 234.88 MB (hbm) | ✓ 0.011×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax-assoc` | ok | 24.23 ms / 24.48 ms | 10.583 s | 385.94 MB (hbm) | ✓ 0.051×tol | 12.49x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax-fft` | ok | 1.93 ms / 1.95 ms | 250.31 ms | 234.88 MB (hbm) | ✓ 0.011×tol | 0.99x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `nitrix-jax-scan` | ok | 6.186 s / 6.537 s | 7.112 s | 352.32 MB (hbm) | ✓ 0.001×tol | 3188.82x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=64,obs=65536,order=8 | `scipy.signal.sosfilt` | ok | 46.67 ms / 49.78 ms | 56.72 ms | 16.78 MB (hbm) | ✓ 0.032×tol | 24.06x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

