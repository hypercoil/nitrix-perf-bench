# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 645ce27d898f29997eff5632fb251170ec24d312 | bench: 126d3c00c62903f6e77922b34c777532d2a08b63
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T02:25:35.549340+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cosine_kernel | jax-cpu | n=2048,d=64 | `cupy.cosine_kernel` | skipped | — | — | — | — | — |
| cosine_kernel | jax-cpu | n=2048,d=64 | `nitrix-jax` | ok | 3.99 ms / 8.40 ms | 166.95 ms | 865 MB (rss) | ✓ 0.00089×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cpu | n=2048,d=64 | `sklearn.cosine_similarity` | ok | 11.94 ms / 14.00 ms | 17.69 ms | 865 MB (rss) | ✓ 0.00089×tol | 2.99x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=2048,d=64 | `cupy.cosine_kernel` | ok | 232.3 µs / 238.1 µs | 104.42 ms | 0.52 MB (hbm) | ✓ 0.00078×tol | 1.34x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=2048,d=64 | `nitrix-jax` | ok | 173.6 µs / 185.8 µs | 258.33 ms | 105.38 MB (hbm) | ✓ 0.00084×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=2048,d=64 | `sklearn.cosine_similarity` | ok | 25.19 ms / 35.81 ms | 71.42 ms | 0.52 MB (hbm) | ✓ 0.00089×tol | 145.12x vs nitrix-jax |
| cosine_kernel | jax-cpu | n=4096,d=64 | `cupy.cosine_kernel` | skipped | — | — | — | — | — |
| cosine_kernel | jax-cpu | n=4096,d=64 | `nitrix-jax` | ok | 36.00 ms / 46.22 ms | 200.05 ms | 924 MB (rss) | ✓ 0.0011×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cpu | n=4096,d=64 | `sklearn.cosine_similarity` | ok | 78.57 ms / 98.29 ms | 73.88 ms | 865 MB (rss) | ✓ 0.0011×tol | 2.18x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=4096,d=64 | `cupy.cosine_kernel` | ok | 1.44 ms / 1.45 ms | 104.13 ms | 1.05 MB (hbm) | ✓ 0.0011×tol | 3.45x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=4096,d=64 | `nitrix-jax` | ok | 417.1 µs / 421.2 µs | 509.53 ms | 206.57 MB (hbm) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=4096,d=64 | `sklearn.cosine_similarity` | ok | 68.23 ms / 70.32 ms | 63.12 ms | 1.05 MB (hbm) | ✓ 0.0011×tol | 163.58x vs nitrix-jax |
| cosine_kernel | jax-cpu | n=512,d=64 | `cupy.cosine_kernel` | skipped | — | — | — | — | — |
| cosine_kernel | jax-cpu | n=512,d=64 | `nitrix-jax` | ok | 271.7 µs / 275.3 µs | 124.63 ms | 865 MB (rss) | ✓ 0.00074×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cpu | n=512,d=64 | `sklearn.cosine_similarity` | ok | 705.7 µs / 8.00 ms | 1.06 ms | 865 MB (rss) | ✓ 0.00067×tol | 2.60x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=512,d=64 | `cupy.cosine_kernel` | ok | 82.8 µs / 84.9 µs | 394.71 ms | 0.13 MB (hbm) | ✓ 0.00066×tol | 0.70x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=512,d=64 | `nitrix-jax` | ok | 118.3 µs / 133.3 µs | 323.41 ms | 74.45 MB (hbm) | ✓ 0.00069×tol | 1.00x vs nitrix-jax |
| cosine_kernel | jax-cuda12 | n=512,d=64 | `sklearn.cosine_similarity` | ok | 679.0 µs / 684.6 µs | 1.30 ms | 0.13 MB (hbm) | ✓ 0.00067×tol | 5.74x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

