# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: a07697668f07841f08dd5fda745a5c75072def6d | bench: 02d1a5d97bf9a38189449744ade4dd1e7f0d8df9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T06:26:40.735660+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| correlation_ratio | jax-cuda12 | shape=[128, 128, 128] | `cupy.cr` | ok | 7.92 ms / 7.93 ms | 228.62 ms | 16.78 MB (hbm) | ✓ 4.1e-05×tol | 2.08x vs nitrix-jax |
| correlation_ratio | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 3.81 ms / 3.82 ms | 606.23 ms | 100.66 MB (hbm) | ✓ 0.011×tol | 1.00x vs nitrix-jax |
| correlation_ratio | jax-cuda12 | shape=[128, 128, 128] | `numpy.cr` | ok | 79.53 ms / 80.55 ms | 80.65 ms | 16.78 MB (hbm) | ✓ 0×tol | 20.87x vs nitrix-jax |
| correlation_ratio | jax-cuda12 | shape=[64, 64, 64] | `cupy.cr` | ok | 1.54 ms / 1.56 ms | 1.356 s | 2.10 MB (hbm) | ✓ 0.00012×tol | 3.00x vs nitrix-jax |
| correlation_ratio | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 513.2 µs / 525.5 µs | 559.88 ms | 85.98 MB (hbm) | ✓ 0.0045×tol | 1.00x vs nitrix-jax |
| correlation_ratio | jax-cuda12 | shape=[64, 64, 64] | `numpy.cr` | ok | 7.98 ms / 8.12 ms | 8.28 ms | 2.10 MB (hbm) | ✓ 0×tol | 15.54x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

