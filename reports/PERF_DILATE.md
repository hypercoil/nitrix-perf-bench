# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: ddf01cd928b0c356b2c7a1ac7c089eca3207435e
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T04:27:33.459613+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| dilate | jax-cpu | shape=[256, 256],size=3 | `nitrix-jax` | ok | 694.2 µs / 737.8 µs | 152.98 ms | 231 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[256, 256],size=3 | `scipy.ndimage.grey_dilation` | ok | 1.32 ms / 1.33 ms | 1.35 ms | 172 MB (rss) | ✓ 0×tol | 1.90x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `nitrix-jax` | ok | 76.8 µs / 77.5 µs | 141.61 ms | 224 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| dilate | jax-cpu | shape=[64, 64],size=3 | `scipy.ndimage.grey_dilation` | ok | 74.9 µs / 78.3 µs | 114.9 µs | 167 MB (rss) | ✓ 0×tol | 0.98x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

