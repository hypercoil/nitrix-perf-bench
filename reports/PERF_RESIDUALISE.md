# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e6787f80311a5b43a0810179cc6ebc802a92a3d3 | bench: c2c8b19f80c6fa804b4d2065042e2301e3c4c0d4
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-24T03:24:54.718194+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| residualise | jax-cpu | V=1000,N=400,K=24 | `nitrix-jax` | ok | 1.14 ms / 1.41 ms | 469.93 ms | 1525 MB (rss) | ✓ 0.0038×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=1000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 9.25 ms / 14.57 ms | 11.49 ms | 1525 MB (rss) | ✓ 0.0017×tol | 8.09x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `nitrix-jax` | ok | 16.08 ms / 22.65 ms | 455.05 ms | 1525 MB (rss) | ✓ 0.0061×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=10000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 162.69 ms / 181.74 ms | 160.84 ms | 1525 MB (rss) | ✓ 0.0022×tol | 10.12x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `nitrix-jax` | ok | 228.82 ms / 244.86 ms | 605.66 ms | 3013 MB (rss) | ✓ 0.0071×tol | 1.00x vs nitrix-jax |
| residualise | jax-cpu | V=100000,N=400,K=24 | `numpy.linalg.lstsq` | ok | 2.289 s / 2.358 s | 2.228 s | 2839 MB (rss) | ✓ 0.0025×tol | 10.01x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

