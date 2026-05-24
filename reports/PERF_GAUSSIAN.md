# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: ddf01cd928b0c356b2c7a1ac7c089eca3207435e
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T04:27:22.413375+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `nitrix-jax` | ok | 3.13 ms / 3.77 ms | 231.16 ms | 256 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[256, 256],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 852.3 µs / 867.9 µs | 884.5 µs | 173 MB (rss) | ✓ 0.00021×tol | 0.27x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `nitrix-jax` | ok | 24.72 ms / 31.76 ms | 206.57 ms | 328 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 10.03 ms / 11.69 ms | 10.80 ms | 182 MB (rss) | ✓ 0.00016×tol | 0.41x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `nitrix-jax` | ok | 462.6 µs / 480.1 µs | 124.03 ms | 225 MB (rss) | ✓ 0.00047×tol | 1.00x vs nitrix-jax |
| gaussian | jax-cpu | shape=[64, 64],sigma=1.5 | `scipy.ndimage.gaussian_filter` | ok | 78.9 µs / 83.2 µs | 107.1 µs | 173 MB (rss) | ✓ 0.00019×tol | 0.17x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

