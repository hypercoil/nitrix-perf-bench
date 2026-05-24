# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: c2c8b19f80c6fa804b4d2065042e2301e3c4c0d4
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T03:23:59.848212+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| cov | jax-cpu | c=2000,n_obs=1000 | `nitrix-jax` | ok | 48.68 ms / 51.76 ms | 173.77 ms | 533 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=2000,n_obs=1000 | `numpy.cov` | ok | 73.95 ms / 89.08 ms | 109.92 ms | 403 MB (rss) | ✓ 0×tol | 1.52x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `nitrix-jax` | ok | 68.7 µs / 69.2 µs | 144.64 ms | 239 MB (rss) | ✓ 0.00053×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=50,n_obs=500 | `numpy.cov` | ok | 126.0 µs / 127.7 µs | 166.8 µs | 239 MB (rss) | ✓ 0×tol | 1.83x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `nitrix-jax` | ok | 6.03 ms / 6.74 ms | 140.84 ms | 240 MB (rss) | ✓ 0.00076×tol | 1.00x vs nitrix-jax |
| cov | jax-cpu | c=500,n_obs=2000 | `numpy.cov` | ok | 8.80 ms / 15.59 ms | 13.11 ms | 239 MB (rss) | ✓ 0×tol | 1.46x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

