# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 269e36a39ca6438c2cf0e3b8c11a599f6ba7d487 | bench: b2f67826811ec859d362bdd636fb9690719344f7
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-06T06:05:12.873275+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| distance_transform_chamfer | jax-cpu | shape=[128, 128] | `nitrix-jax` | ok | 29.84 ms / 36.20 ms | 276.96 ms | 703 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cpu | shape=[128, 128] | `scipy.ndimage.distance_transform_cdt` | ok | 175.4 µs / 186.2 µs | 201.0 µs | 703 MB (rss) | ✓ 0×tol | 0.01x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[128, 128] | `nitrix-jax` | ok | 5.02 ms / 8.72 ms | 737.65 ms | 68.35 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[128, 128] | `scipy.ndimage.distance_transform_cdt` | ok | 330.3 µs / 337.0 µs | 362.8 µs | 0.07 MB (hbm) | ✓ 0×tol | 0.07x vs nitrix-jax |
| distance_transform_chamfer | jax-cpu | shape=[64, 64, 64] | `nitrix-jax` | ok | 1.810 s / 1.992 s | 2.206 s | 703 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cpu | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_cdt` | ok | 4.70 ms / 5.90 ms | 35.87 ms | 703 MB (rss) | ✓ 0×tol | 0.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[64, 64, 64] | `nitrix-jax` | ok | 23.20 ms / 23.30 ms | 985.29 ms | 135.27 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[64, 64, 64] | `scipy.ndimage.distance_transform_cdt` | ok | 5.64 ms / 9.23 ms | 4.56 ms | 1.05 MB (hbm) | ✓ 0×tol | 0.24x vs nitrix-jax |
| distance_transform_chamfer | jax-cpu | shape=[64, 64] | `nitrix-jax` | ok | 5.15 ms / 5.44 ms | 424.92 ms | 703 MB (rss) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cpu | shape=[64, 64] | `scipy.ndimage.distance_transform_cdt` | ok | 86.9 µs / 88.1 µs | 115.4 µs | 703 MB (rss) | ✓ 0×tol | 0.02x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[64, 64] | `nitrix-jax` | ok | 3.20 ms / 3.25 ms | 858.97 ms | 33.87 MB (hbm) | ✓ 0×tol | 1.00x vs nitrix-jax |
| distance_transform_chamfer | jax-cuda12 | shape=[64, 64] | `scipy.ndimage.distance_transform_cdt` | ok | 115.8 µs / 124.1 µs | 172.2 µs | 0.02 MB (hbm) | ✓ 0×tol | 0.04x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

