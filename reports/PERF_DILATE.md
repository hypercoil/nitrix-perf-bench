# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: 44def4b7ce5c1f37844a65f1545ddc4ba9281c5b
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T22:36:41.935810+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cpu | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 788.2 µs / 868.5 µs | 143.62 ms | 698 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.22 ms / 1.24 ms | 1.25 ms | 698 MB (rss) | ✓ 0×tol | 1.55x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],size=3 | `simpleitk.GrayscaleDilate` | ok | 4.41 ms / 5.15 ms | 70.24 ms | 698 MB (rss) | ✓ 0×tol | 5.59x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 118.7 µs / 125.7 µs | 147.81 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.80x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `nitrix-jax` | ok | 149.1 µs / 155.9 µs | 547.80 ms | 72.09 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.21 ms / 1.22 ms | 1.25 ms | 0.26 MB (hbm) | ✓ 0×tol | 8.11x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[256, 256],size=3 | `simpleitk.GrayscaleDilate` | ok | 3.26 ms / 3.29 ms | 61.01 ms | 0.26 MB (hbm) | ✓ 0×tol | 21.84x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_dilation` | skipped | — | — | — | — | — |
| dilate | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 84.1 µs / 85.2 µs | 116.90 ms | 698 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 68.7 µs / 92.1 µs | 134.6 µs | 698 MB (rss) | ✓ 0×tol | 0.82x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `simpleitk.GrayscaleDilate` | ok | 1.18 ms / 1.24 ms | 56.44 ms | 698 MB (rss) | ✓ 0×tol | 14.07x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `cupyx.scipy.ndimage.grey_dilation` | ok | 142.6 µs / 146.8 µs | 175.21 ms | 0.02 MB (hbm) | ✓ 0×tol | 0.99x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `nitrix-jax` | ok | 144.0 µs / 146.0 µs | 506.75 ms | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 59.6 µs / 67.3 µs | 112.9 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.41x vs nitrix-jax |
| dilate | jax-cuda12 | shape=[64, 64],size=3 | `simpleitk.GrayscaleDilate` | ok | 1.15 ms / 1.26 ms | 61.53 ms | 0.02 MB (hbm) | ✓ 0×tol | 8.01x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

