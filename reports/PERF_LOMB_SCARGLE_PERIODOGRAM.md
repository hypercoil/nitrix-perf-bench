# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: fa0a230c55f2a768bb14670d07c7d23c6f102590
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T03:13:51.167048+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| lomb_scargle_periodogram | jax-cpu | obs=2048 | `cupyx.scipy.signal.lombscargle` | skipped | — | — | — | — | — |
| lomb_scargle_periodogram | jax-cpu | obs=2048 | `nitrix-jax` | ok | 56.54 ms / 61.60 ms | 263.24 ms | 1216 MB (rss) | ✓ 0.46×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cpu | obs=2048 | `scipy.signal.lombscargle` | ok | 207.51 ms / 223.81 ms | 246.85 ms | 1216 MB (rss) | ✓ 0×tol | 3.67x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=2048 | `cupyx.scipy.signal.lombscargle` | ok | 16.26 ms / 16.29 ms | 977.31 ms | 33.64 MB (hbm) | ✓ 9.4e-10×tol | 108.67x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=2048 | `nitrix-jax` | ok | 149.6 µs / 153.1 µs | 1.353 s | 33.64 MB (hbm) | ✓ 0.43×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=2048 | `scipy.signal.lombscargle` | ok | 221.74 ms / 279.75 ms | 299.12 ms | 33.64 MB (hbm) | ✓ 0×tol | 1481.81x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cpu | obs=4096 | `cupyx.scipy.signal.lombscargle` | skipped | — | — | — | — | — |
| lomb_scargle_periodogram | jax-cpu | obs=4096 | `nitrix-jax` | ok | 231.17 ms / 384.13 ms | 449.20 ms | 1216 MB (rss) | ✓ 0.75×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cpu | obs=4096 | `scipy.signal.lombscargle` | ok | 741.30 ms / 827.17 ms | 767.57 ms | 1216 MB (rss) | ✓ 0×tol | 3.21x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=4096 | `cupyx.scipy.signal.lombscargle` | ok | 32.15 ms / 32.20 ms | 714.08 ms | 134.39 MB (hbm) | ✓ 1.4e-09×tol | 147.18x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=4096 | `nitrix-jax` | ok | 218.4 µs / 220.9 µs | 899.18 ms | 134.39 MB (hbm) | ✓ 0.83×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=4096 | `scipy.signal.lombscargle` | ok | 729.64 ms / 739.31 ms | 739.08 ms | 134.39 MB (hbm) | ✓ 0×tol | 3340.64x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cpu | obs=512 | `cupyx.scipy.signal.lombscargle` | skipped | — | — | — | — | — |
| lomb_scargle_periodogram | jax-cpu | obs=512 | `nitrix-jax` | ok | 4.16 ms / 4.49 ms | 202.64 ms | 1216 MB (rss) | ✓ 0.19×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cpu | obs=512 | `scipy.signal.lombscargle` | ok | 6.26 ms / 6.30 ms | 6.50 ms | 1216 MB (rss) | ✓ 0×tol | 1.50x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=512 | `cupyx.scipy.signal.lombscargle` | ok | 1.37 ms / 1.38 ms | 2.017 s | 34.09 MB (hbm) | ✓ 7.6e-11×tol | 12.16x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=512 | `nitrix-jax` | ok | 112.7 µs / 114.3 µs | 960.89 ms | 34.09 MB (hbm) | ✓ 0.19×tol | 1.00x vs nitrix-jax |
| lomb_scargle_periodogram | jax-cuda12 | obs=512 | `scipy.signal.lombscargle` | ok | 9.24 ms / 9.81 ms | 9.81 ms | 34.09 MB (hbm) | ✓ 0×tol | 82.03x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

