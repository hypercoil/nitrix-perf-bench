# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: fcbe94d384b8576bb7f2f515756b521236682272 | bench: 7159b33d2e4fe9f95c9156aa9ceffdafd949591d
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-01T23:48:34.486014+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| rbf_kernel | jax-cpu | n=2048,d=64 | `cupy.rbf_kernel` | skipped | — | — | — | — | — |
| rbf_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 7.87 ms / 15.99 ms | 179.40 ms | 927 MB (rss) | ✓ 0.00087×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cpu | n=2048,d=64 | `sklearn.rbf_kernel` | ok | 32.88 ms / 50.61 ms | 52.13 ms | 927 MB (rss) | ✓ 0.00017×tol | 4.18x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=2048,d=64 | `cupy.rbf_kernel` | ok | 587.3 µs / 598.1 µs | 111.65 ms | 0.52 MB (hbm) | ✓ 0.00068×tol | 3.20x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 183.6 µs / 186.2 µs | 567.79 ms | 105.38 MB (hbm) | ✓ 0.00087×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.rbf_kernel` | ok | 54.44 ms / 92.22 ms | 112.46 ms | 0.52 MB (hbm) | ✓ 0.00017×tol | 296.54x vs nitrix-jax |
| rbf_kernel | jax-cpu | n=4096,d=64 | `cupy.rbf_kernel` | skipped | — | — | — | — | — |
| rbf_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 34.14 ms / 65.05 ms | 436.02 ms | 927 MB (rss) | ✓ 0.00087×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cpu | n=4096,d=64 | `sklearn.rbf_kernel` | ok | 199.74 ms / 270.64 ms | 232.16 ms | 927 MB (rss) | ✓ 0.00017×tol | 5.85x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=4096,d=64 | `cupy.rbf_kernel` | ok | 3.86 ms / 3.88 ms | 140.67 ms | 1.05 MB (hbm) | ✓ 0.00072×tol | 4.09x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 942.4 µs / 949.6 µs | 740.19 ms | 206.57 MB (hbm) | ✓ 0.00087×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.rbf_kernel` | ok | 174.44 ms / 175.75 ms | 179.65 ms | 1.05 MB (hbm) | ✓ 0.00017×tol | 185.10x vs nitrix-jax |
| rbf_kernel | jax-cpu | n=512,d=64 | `cupy.rbf_kernel` | skipped | — | — | — | — | — |
| rbf_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 404.3 µs / 768.4 µs | 146.23 ms | 927 MB (rss) | ✓ 0.00065×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cpu | n=512,d=64 | `sklearn.rbf_kernel` | ok | 2.14 ms / 3.54 ms | 5.28 ms | 927 MB (rss) | ✓ 0.00015×tol | 5.29x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=512,d=64 | `cupy.rbf_kernel` | ok | 124.7 µs / 135.1 µs | 219.19 ms | 0.13 MB (hbm) | ✓ 0.0005×tol | 1.07x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 116.9 µs / 121.7 µs | 344.78 ms | 74.45 MB (hbm) | ✓ 0.0006×tol | 1.00x vs nitrix-jax |
| rbf_kernel | jax-cuda12 | n=512,d=64 | `sklearn.rbf_kernel` | ok | 1.91 ms / 5.18 ms | 19.33 ms | 0.13 MB (hbm) | ✓ 0.00015×tol | 16.34x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

