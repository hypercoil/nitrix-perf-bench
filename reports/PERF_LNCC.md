# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: a07697668f07841f08dd5fda745a5c75072def6d | bench: 02d1a5d97bf9a38189449744ade4dd1e7f0d8df9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T06:22:59.593028+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| lncc | jax-cuda12 | shape=[128, 128, 128] | `cupy.lncc` | ok | 5.64 ms / 5.82 ms | 268.83 ms | 16.78 MB (hbm) | ✓ 1e-13×tol | 3.49x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 1.61 ms / 1.65 ms | 1.538 s | 8758.24 MB (hbm) | ✓ 4.3e-05×tol | 1.00x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[128, 128, 128] | `numpy.lncc` | ok | 315.69 ms / 319.77 ms | 320.23 ms | 16.78 MB (hbm) | ✓ 0×tol | 195.63x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[128, 128, 128] | `simpleitk.ANTSNeighborhoodCorrelation` | ok | 4.082 s / 4.112 s | 4.597 s | 16.78 MB (hbm) | ≈ 1.8e+03×tol | 2529.57x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[64, 64, 64] | `cupy.lncc` | ok | 892.9 µs / 905.6 µs | 12.062 s | 2.10 MB (hbm) | ✓ 0×tol | 2.77x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 322.8 µs / 332.0 µs | 5.614 s | 616.70 MB (hbm) | ✓ 7e-06×tol | 1.00x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[64, 64, 64] | `numpy.lncc` | ok | 32.04 ms / 32.63 ms | 32.36 ms | 2.10 MB (hbm) | ✓ 0×tol | 99.25x vs nitrix-jax |
| lncc | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.ANTSNeighborhoodCorrelation` | ok | 501.89 ms / 506.00 ms | 572.38 ms | 2.10 MB (hbm) | ≈ 1.8e+03×tol | 1554.62x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

