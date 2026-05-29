# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 5229f3d4ac94637c1ce7878c660330f3d55e0a8d
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T05:44:41.288109+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| analytic_signal | jax-cpu | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| analytic_signal | jax-cpu | n_sig=2048,t=2048 | `nitrix-jax` | ok | 27.22 ms / 32.95 ms | 119.14 ms | 1364 MB (rss) | ✓ 0.0053×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cpu | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 56.52 ms / 57.20 ms | 54.65 ms | 1364 MB (rss) | ✓ 0.0058×tol | 2.08x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | ok | 792.9 µs / 802.1 µs | 412.75 ms | 16.78 MB (hbm) | ✓ 0.0078×tol | 1.11x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=2048,t=2048 | `nitrix-jax` | ok | 716.6 µs / 722.3 µs | 239.31 ms | 117.44 MB (hbm) | ✓ 0.0078×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 55.67 ms / 55.94 ms | 53.72 ms | 16.78 MB (hbm) | ✓ 0.0058×tol | 77.70x vs nitrix-jax |
| analytic_signal | jax-cpu | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| analytic_signal | jax-cpu | n_sig=4096,t=4096 | `nitrix-jax` | ok | 153.47 ms / 200.99 ms | 225.65 ms | 1364 MB (rss) | ✓ 0.006×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cpu | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 229.59 ms / 233.93 ms | 218.48 ms | 1364 MB (rss) | ✓ 0.0069×tol | 1.50x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | ok | 5.35 ms / 5.37 ms | 369.23 ms | 67.11 MB (hbm) | ✓ 0.0088×tol | 0.98x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=4096,t=4096 | `nitrix-jax` | ok | 5.49 ms / 5.51 ms | 218.53 ms | 469.76 MB (hbm) | ✓ 0.0088×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 229.60 ms / 230.64 ms | 221.95 ms | 67.11 MB (hbm) | ✓ 0.0069×tol | 41.85x vs nitrix-jax |
| analytic_signal | jax-cpu | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| analytic_signal | jax-cpu | n_sig=512,t=1024 | `nitrix-jax` | ok | 2.31 ms / 2.35 ms | 90.68 ms | 1364 MB (rss) | ✓ 0.0047×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cpu | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 2.76 ms / 2.84 ms | 5.71 ms | 1364 MB (rss) | ✓ 0.0048×tol | 1.20x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | ok | 172.3 µs / 178.4 µs | 2.065 s | 2.10 MB (hbm) | ✓ 0.0072×tol | 1.34x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=512,t=1024 | `nitrix-jax` | ok | 128.7 µs / 193.2 µs | 340.42 ms | 14.68 MB (hbm) | ✓ 0.0072×tol | 1.00x vs nitrix-jax |
| analytic_signal | jax-cuda12 | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 2.82 ms / 4.31 ms | 5.96 ms | 2.10 MB (hbm) | ✓ 0.0048×tol | 21.93x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

