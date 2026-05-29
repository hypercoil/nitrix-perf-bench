# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 5229f3d4ac94637c1ce7878c660330f3d55e0a8d
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-29T05:45:37.395267+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| hilbert_transform | jax-cpu | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| hilbert_transform | jax-cpu | n_sig=2048,t=2048 | `nitrix-jax` | ok | 34.15 ms / 45.36 ms | 159.21 ms | 1364 MB (rss) | ✓ 0.0066×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cpu | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 55.80 ms / 57.17 ms | 53.91 ms | 1364 MB (rss) | ✓ 0.0072×tol | 1.63x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=2048,t=2048 | `cupyx.scipy.signal.hilbert` | ok | 795.0 µs / 805.0 µs | 478.11 ms | 16.78 MB (hbm) | ✓ 0.01×tol | 1.03x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=2048,t=2048 | `nitrix-jax` | ok | 771.7 µs / 800.8 µs | 170.33 ms | 117.44 MB (hbm) | ✓ 0.01×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=2048,t=2048 | `scipy.signal.hilbert` | ok | 56.00 ms / 57.39 ms | 54.08 ms | 16.78 MB (hbm) | ✓ 0.0072×tol | 72.58x vs nitrix-jax |
| hilbert_transform | jax-cpu | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| hilbert_transform | jax-cpu | n_sig=4096,t=4096 | `nitrix-jax` | ok | 140.11 ms / 183.85 ms | 264.70 ms | 1364 MB (rss) | ✓ 0.0079×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cpu | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 234.45 ms / 261.02 ms | 308.36 ms | 1364 MB (rss) | ✓ 0.0087×tol | 1.67x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=4096,t=4096 | `cupyx.scipy.signal.hilbert` | ok | 5.34 ms / 5.36 ms | 371.60 ms | 67.11 MB (hbm) | ✓ 0.012×tol | 0.84x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=4096,t=4096 | `nitrix-jax` | ok | 6.32 ms / 6.39 ms | 171.85 ms | 469.76 MB (hbm) | ✓ 0.012×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=4096,t=4096 | `scipy.signal.hilbert` | ok | 223.70 ms / 224.98 ms | 220.61 ms | 67.11 MB (hbm) | ✓ 0.0087×tol | 35.38x vs nitrix-jax |
| hilbert_transform | jax-cpu | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | skipped | — | — | — | — | — |
| hilbert_transform | jax-cpu | n_sig=512,t=1024 | `nitrix-jax` | ok | 2.42 ms / 2.44 ms | 89.17 ms | 1364 MB (rss) | ✓ 0.0058×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cpu | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 2.77 ms / 3.92 ms | 4.25 ms | 1364 MB (rss) | ✓ 0.0058×tol | 1.14x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=512,t=1024 | `cupyx.scipy.signal.hilbert` | ok | 179.9 µs / 188.0 µs | 342.59 ms | 2.10 MB (hbm) | ✓ 0.0085×tol | 1.23x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=512,t=1024 | `nitrix-jax` | ok | 146.4 µs / 147.9 µs | 153.46 ms | 14.68 MB (hbm) | ✓ 0.0085×tol | 1.00x vs nitrix-jax |
| hilbert_transform | jax-cuda12 | n_sig=512,t=1024 | `scipy.signal.hilbert` | ok | 2.93 ms / 5.13 ms | 7.37 ms | 2.10 MB (hbm) | ✓ 0.0058×tol | 20.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

