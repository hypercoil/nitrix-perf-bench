# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: a07697668f07841f08dd5fda745a5c75072def6d | bench: 02d1a5d97bf9a38189449744ade4dd1e7f0d8df9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T06:09:55.597022+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| ncc | jax-cuda12 | shape=[128, 128, 128] | `cupy.ncc` | ok | 236.2 µs / 252.6 µs | 169.66 ms | 16.78 MB (hbm) | ✓ 1.5e-05×tol | 2.03x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 116.1 µs / 147.7 µs | 338.55 ms | 50.33 MB (hbm) | ✓ 9.4e-05×tol | 1.00x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[128, 128, 128] | `numpy.ncc` | ok | 24.99 ms / 25.13 ms | 25.09 ms | 16.78 MB (hbm) | ✓ 0×tol | 215.17x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[128, 128, 128] | `simpleitk.Correlation` | ok | 272.16 ms / 299.52 ms | 358.20 ms | 16.78 MB (hbm) | ≈ 1.8e+03×tol | 2343.71x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[64, 64, 64] | `cupy.ncc` | ok | 243.3 µs / 252.0 µs | 2.031 s | 2.10 MB (hbm) | ✓ 9.7e-05×tol | 2.02x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 120.4 µs / 126.0 µs | 312.14 ms | 35.65 MB (hbm) | ✓ 4.3e-05×tol | 1.00x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[64, 64, 64] | `numpy.ncc` | ok | 2.37 ms / 2.40 ms | 2.46 ms | 2.10 MB (hbm) | ✓ 0×tol | 19.66x vs nitrix-jax |
| ncc | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.Correlation` | ok | 31.46 ms / 34.49 ms | 101.58 ms | 2.10 MB (hbm) | ≈ 1.8e+03×tol | 261.30x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

