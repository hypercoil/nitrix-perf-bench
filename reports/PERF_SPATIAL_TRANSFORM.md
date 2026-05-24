# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: 8c20dab9d447d47f4d1efaab756cd71f9e611b48
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T04:58:34.045516+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| spatial_transform | jax-cpu | shape=[256, 256] | `nitrix-jax` | ok | 346.0 µs / 413.7 µs | 151.25 ms | 216 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[256, 256] | `scipy.ndimage.map_coordinates` | ok | 2.55 ms / 2.65 ms | 2.61 ms | 172 MB (rss) | ✓ 5.5e-05×tol | 7.37x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `nitrix-jax` | ok | 81.1 µs / 93.5 µs | 153.02 ms | 212 MB (rss) | ✓ 0.00042×tol | 1.00x vs nitrix-jax |
| spatial_transform | jax-cpu | shape=[64, 64] | `scipy.ndimage.map_coordinates` | ok | 154.6 µs / 194.2 µs | 172.8 µs | 172 MB (rss) | ✓ 5.2e-05×tol | 1.91x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

