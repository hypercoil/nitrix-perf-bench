# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: a07697668f07841f08dd5fda745a5c75072def6d | bench: 02d1a5d97bf9a38189449744ade4dd1e7f0d8df9
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T06:25:15.215958+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| mutual_information | jax-cuda12 | shape=[128, 128, 128] | `cupy.mi` | ok | 4.77 ms / 4.82 ms | 257.04 ms | 16.78 MB (hbm) | ✓ 1.8e-05×tol | 3.53x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[128, 128, 128] | `nitrix-jax` | ok | 1.35 ms / 1.42 ms | 687.74 ms | 83.89 MB (hbm) | ✓ 0.00029×tol | 1.00x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[128, 128, 128] | `numpy.mi` | ok | 292.03 ms / 294.05 ms | 294.04 ms | 16.78 MB (hbm) | ✓ 0×tol | 216.23x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[128, 128, 128] | `simpleitk.MattesMI` | ok | 247.05 ms / 252.31 ms | 314.92 ms | 16.78 MB (hbm) | ≈ 1.9e+03×tol | 182.92x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[128, 128, 128] | `sklearn.mutual_info` | ok | 340.69 ms / 345.49 ms | 341.80 ms | 16.78 MB (hbm) | ≈ 2.3e+02×tol | 252.25x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[64, 64, 64] | `cupy.mi` | ok | 1.62 ms / 1.63 ms | 19.572 s | 2.10 MB (hbm) | ✓ 3.4e-05×tol | 5.84x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 277.5 µs / 281.2 µs | 671.67 ms | 69.21 MB (hbm) | ✓ 4.8e-05×tol | 1.00x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[64, 64, 64] | `numpy.mi` | ok | 23.61 ms / 24.01 ms | 23.86 ms | 2.10 MB (hbm) | ✓ 0×tol | 85.08x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[64, 64, 64] | `simpleitk.MattesMI` | ok | 29.11 ms / 30.01 ms | 96.43 ms | 2.10 MB (hbm) | ≈ 1.9e+03×tol | 104.89x vs nitrix-jax |
| mutual_information | jax-cuda12 | shape=[64, 64, 64] | `sklearn.mutual_info` | ok | 37.40 ms / 37.70 ms | 37.99 ms | 2.10 MB (hbm) | ≈ 1.9e+02×tol | 134.74x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

