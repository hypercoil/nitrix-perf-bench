# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 00e7282111c5ee516813bbf6f41e4616cbaff125 | bench: 3e097324252de4a21f5cfae21464658bc4f599c4
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-07T20:50:54.393921+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 232.0 µs / 239.1 µs | 234.60 ms | 0.26 MB (hbm) | ✓ 0×tol | 2.42x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `nitrix-jax` | ok | 96.0 µs / 100.9 µs | 118.40 ms | 1.05 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 2.21 ms / 2.23 ms | 2.28 ms | 0.26 MB (hbm) | ✓ 0×tol | 23.05x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 535.0 µs / 559.3 µs | 201.31 ms | 0.26 MB (hbm) | ✓ 0×tol | 0.85x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `nitrix-jax` | ok | 631.8 µs / 636.6 µs | 848.76 ms | 93.06 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[256, 256],se=disk,radius=3 | `scipy.ndimage.grey_closing` | ok | 2.72 ms / 2.84 ms | 2.85 ms | 0.26 MB (hbm) | ✓ 0×tol | 4.31x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `cupyx.scipy.ndimage.grey_closing` | ok | 554.9 µs / 578.7 µs | 201.89 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.09x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `nitrix-jax` | ok | 6.02 ms / 6.12 ms | 841.82 ms | 336.59 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=ball,radius=2 | `scipy.ndimage.grey_closing` | ok | 12.10 ms / 12.34 ms | 12.22 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.01x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `cupyx.scipy.ndimage.grey_closing` | ok | 320.8 µs / 340.0 µs | 172.01 ms | 1.05 MB (hbm) | ✓ 0×tol | 2.73x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `nitrix-jax` | ok | 117.6 µs / 123.3 µs | 141.84 ms | 4.19 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| close | jax-cuda12 | shape=[64, 64, 64],se=box,size=3 | `scipy.ndimage.grey_closing` | ok | 14.61 ms / 14.66 ms | 14.67 ms | 1.05 MB (hbm) | ✓ 0×tol | 124.21x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

