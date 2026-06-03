# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 8259a204821af34c01e8092f9cf37275f388413d | bench: ed373831de10021fb421d6197a04918a14402586
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T00:23:32.446458+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sosfilt | jax-cpu | channels=1024,obs=4096 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=1024,obs=4096 | `nitrix-jax` | ok | 69.05 ms / 75.04 ms | 303.62 ms | 1163 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=1024,obs=4096 | `nitrix-jax-assoc` | ok | 657.42 ms / 687.25 ms | 2.995 s | 1797 MB (rss) | ✓ 0.0084×tol | 9.52x vs nitrix-jax |
| sosfilt | jax-cpu | channels=1024,obs=4096 | `nitrix-jax-scan` | ok | 66.18 ms / 70.67 ms | 320.35 ms | 1163 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cpu | channels=1024,obs=4096 | `scipy.signal.sosfilt` | ok | 26.68 ms / 28.14 ms | 33.87 ms | 1163 MB (rss) | ✓ 0.0026×tol | 0.39x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=1024,obs=4096 | `cupyx.scipy.signal.sosfilt` | ok | 1.75 ms / 1.77 ms | 2.708 s | 16.78 MB (hbm) | ✓ 0.019×tol | 0.01x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=1024,obs=4096 | `nitrix-jax` | ok | 169.38 ms / 279.73 ms | 703.60 ms | 218.10 MB (hbm) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=1024,obs=4096 | `nitrix-jax-assoc` | ok | 20.31 ms / 20.49 ms | 12.056 s | 553.65 MB (hbm) | ✓ 0.0041×tol | 0.12x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=1024,obs=4096 | `nitrix-jax-scan` | ok | 168.37 ms / 263.26 ms | 702.96 ms | 218.10 MB (hbm) | ✓ 0.00012×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cuda12 | channels=1024,obs=4096 | `scipy.signal.sosfilt` | ok | 26.77 ms / 27.86 ms | 33.88 ms | 16.78 MB (hbm) | ✓ 0.0026×tol | 0.16x vs nitrix-jax |
| sosfilt | jax-cpu | channels=2048,obs=8192 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=2048,obs=8192 | `nitrix-jax` | ok | 262.11 ms / 283.40 ms | 497.67 ms | 1435 MB (rss) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=2048,obs=8192 | `nitrix-jax-assoc` | ok | 2.599 s / 3.101 s | 6.172 s | 5065 MB (rss) | ✓ 0.0094×tol | 9.92x vs nitrix-jax |
| sosfilt | jax-cpu | channels=2048,obs=8192 | `nitrix-jax-scan` | ok | 267.71 ms / 285.25 ms | 531.73 ms | 1429 MB (rss) | ✓ 0.00013×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cpu | channels=2048,obs=8192 | `scipy.signal.sosfilt` | ok | 148.15 ms / 160.77 ms | 159.15 ms | 1163 MB (rss) | ✓ 0.0026×tol | 0.57x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=2048,obs=8192 | `cupyx.scipy.signal.sosfilt` | ok | 9.61 ms / 10.90 ms | 2.873 s | 67.11 MB (hbm) | ✓ 0.023×tol | 0.03x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=2048,obs=8192 | `nitrix-jax` | ok | 330.73 ms / 441.49 ms | 1.141 s | 872.42 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=2048,obs=8192 | `nitrix-jax-assoc` | ok | 103.01 ms / 103.23 ms | 17.492 s | 1242.07 MB (hbm) | ✓ 0.0041×tol | 0.31x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=2048,obs=8192 | `nitrix-jax-scan` | ok | 289.16 ms / 465.98 ms | 958.89 ms | 872.42 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cuda12 | channels=2048,obs=8192 | `scipy.signal.sosfilt` | ok | 155.39 ms / 190.21 ms | 176.55 ms | 67.11 MB (hbm) | ✓ 0.0026×tol | 0.47x vs nitrix-jax |
| sosfilt | jax-cpu | channels=512,obs=2048 | `cupyx.scipy.signal.sosfilt` | skipped | — | — | — | — | — |
| sosfilt | jax-cpu | channels=512,obs=2048 | `nitrix-jax` | ok | 7.52 ms / 8.18 ms | 223.30 ms | 1163 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cpu | channels=512,obs=2048 | `nitrix-jax-assoc` | ok | 141.34 ms / 145.45 ms | 2.443 s | 1163 MB (rss) | ✓ 0.0075×tol | 18.79x vs nitrix-jax |
| sosfilt | jax-cpu | channels=512,obs=2048 | `nitrix-jax-scan` | ok | 7.55 ms / 7.79 ms | 233.42 ms | 1163 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cpu | channels=512,obs=2048 | `scipy.signal.sosfilt` | ok | 6.64 ms / 6.68 ms | 7.17 ms | 1163 MB (rss) | ✓ 0.0026×tol | 0.88x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=512,obs=2048 | `cupyx.scipy.signal.sosfilt` | ok | 569.4 µs / 586.8 µs | 2.744 s | 4.19 MB (hbm) | ✓ 0.018×tol | 0.01x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=512,obs=2048 | `nitrix-jax` | ok | 98.06 ms / 144.02 ms | 586.57 ms | 54.53 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=512,obs=2048 | `nitrix-jax-assoc` | ok | 2.11 ms / 2.15 ms | 8.511 s | 373.48 MB (hbm) | ✓ 0.0038×tol | 0.02x vs nitrix-jax |
| sosfilt | jax-cuda12 | channels=512,obs=2048 | `nitrix-jax-scan` | ok | 90.16 ms / 140.80 ms | 581.04 ms | 54.53 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax-scan |
| sosfilt | jax-cuda12 | channels=512,obs=2048 | `scipy.signal.sosfilt` | ok | 7.62 ms / 8.31 ms | 8.87 ms | 4.19 MB (hbm) | ✓ 0.0026×tol | 0.08x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

