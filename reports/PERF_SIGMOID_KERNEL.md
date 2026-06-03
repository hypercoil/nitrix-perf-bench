# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 645ce27d898f29997eff5632fb251170ec24d312 | bench: 126d3c00c62903f6e77922b34c777532d2a08b63
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T02:27:15.296398+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sigmoid_kernel | jax-cpu | n=2048,d=64 | `cupy.sigmoid_kernel` | skipped | — | — | — | — | — |
| sigmoid_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 5.00 ms / 8.74 ms | 94.02 ms | 864 MB (rss) | ✓ 0.006×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cpu | n=2048,d=64 | `sklearn.sigmoid_kernel` | ok | 29.52 ms / 40.47 ms | 37.65 ms | 864 MB (rss) | ✓ 0.0059×tol | 5.90x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=2048,d=64 | `cupy.sigmoid_kernel` | ok | 282.1 µs / 284.5 µs | 204.13 ms | 0.52 MB (hbm) | ✓ 0.0059×tol | 1.48x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 190.2 µs / 194.5 µs | 367.53 ms | 105.38 MB (hbm) | ✓ 0.0059×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.sigmoid_kernel` | ok | 34.36 ms / 48.79 ms | 68.63 ms | 0.52 MB (hbm) | ✓ 0.0059×tol | 180.66x vs nitrix-jax |
| sigmoid_kernel | jax-cpu | n=4096,d=64 | `cupy.sigmoid_kernel` | skipped | — | — | — | — | — |
| sigmoid_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 29.68 ms / 47.80 ms | 219.82 ms | 903 MB (rss) | ✓ 0.0064×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cpu | n=4096,d=64 | `sklearn.sigmoid_kernel` | ok | 137.20 ms / 160.75 ms | 149.27 ms | 864 MB (rss) | ✓ 0.0064×tol | 4.62x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=4096,d=64 | `cupy.sigmoid_kernel` | ok | 2.01 ms / 2.02 ms | 176.81 ms | 1.05 MB (hbm) | ✓ 0.0064×tol | 2.12x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 945.7 µs / 980.0 µs | 345.53 ms | 206.57 MB (hbm) | ✓ 0.0064×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.sigmoid_kernel` | ok | 132.33 ms / 132.88 ms | 130.41 ms | 1.05 MB (hbm) | ✓ 0.0064×tol | 139.93x vs nitrix-jax |
| sigmoid_kernel | jax-cpu | n=512,d=64 | `cupy.sigmoid_kernel` | skipped | — | — | — | — | — |
| sigmoid_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 355.0 µs / 388.3 µs | 77.46 ms | 864 MB (rss) | ✓ 0.006×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cpu | n=512,d=64 | `sklearn.sigmoid_kernel` | ok | 1.69 ms / 1.72 ms | 2.02 ms | 864 MB (rss) | ✓ 0.0059×tol | 4.76x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=512,d=64 | `cupy.sigmoid_kernel` | ok | 78.4 µs / 81.3 µs | 425.47 ms | 0.13 MB (hbm) | ✓ 0.0059×tol | 0.70x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 111.9 µs / 122.1 µs | 246.34 ms | 74.45 MB (hbm) | ✓ 0.0059×tol | 1.00x vs nitrix-jax |
| sigmoid_kernel | jax-cuda12 | n=512,d=64 | `sklearn.sigmoid_kernel` | ok | 1.67 ms / 1.72 ms | 17.27 ms | 0.13 MB (hbm) | ✓ 0.0059×tol | 14.95x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

