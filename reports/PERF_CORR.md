# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: c2c8b19f80c6fa804b4d2065042e2301e3c4c0d4
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T03:24:08.134661+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| corr | jax-cpu | n=2000,t=1000 | `nitrix-jax` | ok | 43.90 ms / 49.30 ms | 186.78 ms | 505 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=2000,t=1000 | `numpy.corrcoef` | ok | 72.21 ms / 78.08 ms | 103.28 ms | 403 MB (rss) | ✓ 0×tol | 1.64x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `nitrix-jax` | ok | 66.8 µs / 79.9 µs | 173.55 ms | 239 MB (rss) | ✓ 0.00056×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=50,t=500 | `numpy.corrcoef` | ok | 148.8 µs / 152.9 µs | 211.3 µs | 239 MB (rss) | ✓ 0×tol | 2.23x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `nitrix-jax` | ok | 6.27 ms / 7.95 ms | 146.66 ms | 245 MB (rss) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| corr | jax-cpu | n=500,t=2000 | `numpy.corrcoef` | ok | 9.42 ms / 17.65 ms | 11.97 ms | 239 MB (rss) | ✓ 0×tol | 1.50x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

