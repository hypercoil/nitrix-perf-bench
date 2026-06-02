# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: 44def4b7ce5c1f37844a65f1545ddc4ba9281c5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T22:35:55.769528+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| erode | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 756.1 µs / 838.0 µs | 151.68 ms | 698 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.21 ms / 1.22 ms | 1.25 ms | 698 MB (rss) | ✓ 0×tol | 1.60x vs nitrix-jax |
| erode | jax-cpu | shape=[256, 256],size=3 | `simpleitk.GrayscaleErode` | ok | 4.15 ms / 4.68 ms | 67.68 ms | 698 MB (rss) | ✓ 0×tol | 5.48x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 119.8 µs / 126.0 µs | 143.20 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.77x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 156.6 µs / 158.3 µs | 579.10 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_erosion` | ok | 1.21 ms / 1.22 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 7.74x vs nitrix-jax |
| erode | jax-cuda12 | shape=[256, 256],size=3 | `simpleitk.GrayscaleErode` | ok | 3.26 ms / 3.32 ms | 61.14 ms | 0.26 MB (hbm) | ✓ 0×tol | 20.83x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_erosion` | skipped | — | — | — | — | — |
| erode | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 57.6 µs / 58.6 µs | 113.17 ms | 698 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 41.2 µs / 55.6 µs | 116.1 µs | 698 MB (rss) | ✓ 0×tol | 0.72x vs nitrix-jax |
| erode | jax-cpu | shape=[64, 64],size=3 | `simpleitk.GrayscaleErode` | ok | 1.15 ms / 1.17 ms | 56.70 ms | 698 MB (rss) | ✓ 0×tol | 19.90x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_erosion` | ok | 117.9 µs / 123.7 µs | 556.58 ms | 0.02 MB (hbm) | ✓ 0×tol | 0.76x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 156.1 µs / 185.9 µs | 1.258 s | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_erosion` | ok | 78.5 µs / 85.6 µs | 131.5 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.50x vs nitrix-jax |
| erode | jax-cuda12 | shape=[64, 64],size=3 | `simpleitk.GrayscaleErode` | ok | 1.17 ms / 1.25 ms | 58.12 ms | 0.02 MB (hbm) | ✓ 0×tol | 7.51x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

