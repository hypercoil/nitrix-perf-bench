# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 5229f3d4ac94637c1ce7878c660330f3d55e0a8d
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T05:46:29.293650+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| envelope | jax-cpu | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| envelope | jax-cpu | n_sig=2048,t=2048 | `nitrix-jax` | ok | 26.18 ms / 36.96 ms | 125.81 ms | 1332 MB (rss) | ✓ 0.005×tol | 1.00x vs nitrix-jax |
| envelope | jax-cpu | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 58.88 ms / 63.88 ms | 64.82 ms | 1332 MB (rss) | ✓ 0.0046×tol | 2.25x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | ok | 864.8 µs / 868.8 µs | 392.49 ms | 16.78 MB (hbm) | ✓ 0.0077×tol | 1.13x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=2048,t=2048 | `nitrix-jax` | ok | 765.7 µs / 780.6 µs | 180.85 ms | 117.44 MB (hbm) | ✓ 0.0077×tol | 1.00x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 57.33 ms / 58.71 ms | 67.35 ms | 16.78 MB (hbm) | ✓ 0.0046×tol | 74.87x vs nitrix-jax |
| envelope | jax-cpu | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| envelope | jax-cpu | n_sig=4096,t=4096 | `nitrix-jax` | ok | 147.90 ms / 217.26 ms | 292.56 ms | 1332 MB (rss) | ✓ 0.0053×tol | 1.00x vs nitrix-jax |
| envelope | jax-cpu | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 278.88 ms / 309.08 ms | 275.24 ms | 1332 MB (rss) | ✓ 0.0061×tol | 1.89x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | ok | 6.22 ms / 6.24 ms | 375.56 ms | 67.11 MB (hbm) | ✓ 0.0085×tol | 0.98x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=4096,t=4096 | `nitrix-jax` | ok | 6.34 ms / 6.37 ms | 191.81 ms | 469.76 MB (hbm) | ✓ 0.0085×tol | 1.00x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 275.28 ms / 280.27 ms | 276.95 ms | 67.11 MB (hbm) | ✓ 0.0061×tol | 43.39x vs nitrix-jax |
| envelope | jax-cpu | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| envelope | jax-cpu | n_sig=512,t=1024 | `nitrix-jax` | ok | 1.31 ms / 2.63 ms | 90.24 ms | 1332 MB (rss) | ✓ 0.0037×tol | 1.00x vs nitrix-jax |
| envelope | jax-cpu | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 3.25 ms / 3.31 ms | 6.53 ms | 1332 MB (rss) | ✓ 0.0036×tol | 2.48x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | ok | 181.6 µs / 188.4 µs | 629.68 ms | 2.10 MB (hbm) | ✓ 0.0063×tol | 1.45x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=512,t=1024 | `nitrix-jax` | ok | 125.2 µs / 128.2 µs | 167.93 ms | 14.68 MB (hbm) | ✓ 0.0063×tol | 1.00x vs nitrix-jax |
| envelope | jax-cuda12 | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 3.20 ms / 3.66 ms | 6.29 ms | 2.10 MB (hbm) | ✓ 0.0036×tol | 25.54x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

