# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: fa0a230c55f2a768bb14670d07c7d23c6f102590
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T03:13:01.066207+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| tsconv | jax-cpu | obs=16384,k=63 | `cupyx.scipy.signal.correlate` | skipped | — | — | — | — | — |
| tsconv | jax-cpu | obs=16384,k=63 | `nitrix-jax` | ok | 540.9 µs / 654.4 µs | 85.39 ms | 543 MB (rss) | ✓ 0.0091×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cpu | obs=16384,k=63 | `scipy.signal.correlate` | ok | 432.6 µs / 435.7 µs | 459.0 µs | 506 MB (rss) | ✓ 0.0095×tol | 0.80x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=16384,k=63 | `cupyx.scipy.signal.correlate` | ok | 219.9 µs / 230.2 µs | 473.32 ms | 0.07 MB (hbm) | ✓ 0.054×tol | 1.77x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=16384,k=63 | `nitrix-jax` | ok | 124.4 µs / 125.6 µs | 326.18 ms | 67.31 MB (hbm) | ✓ 0.022×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=16384,k=63 | `scipy.signal.correlate` | ok | 428.9 µs / 434.3 µs | 458.4 µs | 0.07 MB (hbm) | ✓ 0.0095×tol | 3.45x vs nitrix-jax |
| tsconv | jax-cpu | obs=4096,k=15 | `cupyx.scipy.signal.correlate` | skipped | — | — | — | — | — |
| tsconv | jax-cpu | obs=4096,k=15 | `nitrix-jax` | ok | 230.7 µs / 241.9 µs | 81.56 ms | 533 MB (rss) | ✓ 0.0046×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cpu | obs=4096,k=15 | `scipy.signal.correlate` | ok | 64.6 µs / 85.3 µs | 102.7 µs | 506 MB (rss) | ✓ 0.0015×tol | 0.28x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=4096,k=15 | `cupyx.scipy.signal.correlate` | ok | 226.8 µs / 232.5 µs | 2.030 s | 0.02 MB (hbm) | ✓ 0.032×tol | 2.22x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=4096,k=15 | `nitrix-jax` | ok | 102.2 µs / 105.6 µs | 1.433 s | 67.16 MB (hbm) | ✓ 0.0048×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=4096,k=15 | `scipy.signal.correlate` | ok | 99.7 µs / 102.4 µs | 131.5 µs | 0.02 MB (hbm) | ✓ 0.0015×tol | 0.98x vs nitrix-jax |
| tsconv | jax-cpu | obs=65536,k=127 | `cupyx.scipy.signal.correlate` | skipped | — | — | — | — | — |
| tsconv | jax-cpu | obs=65536,k=127 | `nitrix-jax` | ok | 2.21 ms / 2.51 ms | 102.25 ms | 560 MB (rss) | ✓ 0.027×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cpu | obs=65536,k=127 | `scipy.signal.correlate` | ok | 1.87 ms / 2.17 ms | 2.61 ms | 506 MB (rss) | ✓ 0.016×tol | 0.85x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=65536,k=127 | `cupyx.scipy.signal.correlate` | ok | 232.6 µs / 245.9 µs | 358.95 ms | 0.26 MB (hbm) | ✓ 0.11×tol | 1.62x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=65536,k=127 | `nitrix-jax` | ok | 143.6 µs / 144.9 µs | 306.24 ms | 67.90 MB (hbm) | ✓ 0.047×tol | 1.00x vs nitrix-jax |
| tsconv | jax-cuda12 | obs=65536,k=127 | `scipy.signal.correlate` | ok | 1.85 ms / 1.88 ms | 1.89 ms | 0.26 MB (hbm) | ✓ 0.016×tol | 12.88x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

