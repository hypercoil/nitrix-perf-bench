# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 00e7282111c5ee516813bbf6f41e4616cbaff125 | bench: 3e097324252de4a21f5cfae21464658bc4f599c4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T20:45:01.581431+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 233.1 µs / 241.7 µs | 264.27 ms | 0.26 MB (hbm) | ✓ 0×tol | 2.38x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 98.1 µs / 102.0 µs | 125.69 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 2.20 ms / 2.22 ms | 2.24 ms | 0.26 MB (hbm) | ✓ 0×tol | 22.42x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 538.7 µs / 545.8 µs | 209.60 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.85x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 635.3 µs / 643.8 µs | 869.75 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_opening` | ok | 2.73 ms / 2.88 ms | 2.77 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.30x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_opening` | ok | 558.7 µs / 567.1 µs | 210.77 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.09x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 6.01 ms / 6.11 ms | 843.89 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_opening` | ok | 12.04 ms / 12.41 ms | 12.55 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_opening` | ok | 327.8 µs / 333.4 µs | 168.82 ms | 1.05 MB (hbm) | ✓ 0×tol | 3.03x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 108.3 µs / 113.2 µs | 144.96 ms | 4.19 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| open | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_opening` | ok | 14.57 ms / 14.66 ms | 14.84 ms | 1.05 MB (hbm) | ✓ 0×tol | 134.60x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

