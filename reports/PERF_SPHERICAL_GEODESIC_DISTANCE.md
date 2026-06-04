# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: 8171f3faa23e2dd8c88e3ef9a4602a7b487c0b85
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T05:39:15.900306+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| spherical_geodesic_distance | jax-cpu | n=1024 | `cupy.spherical_geodesic_distance` | skipped | — | — | — | — | — |
| spherical_geodesic_distance | jax-cpu | n=1024 | `nitrix-jax` | ok | 10.82 ms / 13.66 ms | 94.86 ms | 768 MB (rss) | ✓ 0.00024×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cpu | n=1024 | `sklearn.haversine` | ok | 33.24 ms / 36.11 ms | 33.66 ms | 768 MB (rss) | ✓ 0.0013×tol | 3.07x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=1024 | `cupy.spherical_geodesic_distance` | ok | 3.68 ms / 3.74 ms | 134.18 ms | 0.01 MB (hbm) | ✓ 0.00026×tol | 34.01x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=1024 | `nitrix-jax` | ok | 108.3 µs / 111.5 µs | 309.45 ms | 41.96 MB (hbm) | ✓ 0.00026×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=1024 | `sklearn.haversine` | ok | 32.91 ms / 32.97 ms | 33.38 ms | 0.01 MB (hbm) | ✓ 0.0013×tol | 303.96x vs nitrix-jax |
| spherical_geodesic_distance | jax-cpu | n=256 | `cupy.spherical_geodesic_distance` | skipped | — | — | — | — | — |
| spherical_geodesic_distance | jax-cpu | n=256 | `nitrix-jax` | ok | 924.3 µs / 982.0 µs | 85.77 ms | 768 MB (rss) | ✓ 0.00021×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cpu | n=256 | `sklearn.haversine` | ok | 2.42 ms / 2.53 ms | 2.69 ms | 768 MB (rss) | ✓ 0.0011×tol | 2.62x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=256 | `cupy.spherical_geodesic_distance` | ok | 400.3 µs / 406.3 µs | 1.199 s | 0.00 MB (hbm) | ✓ 0.0002×tol | 4.29x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=256 | `nitrix-jax` | ok | 93.4 µs / 95.4 µs | 279.61 ms | 34.08 MB (hbm) | ✓ 0.0002×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=256 | `sklearn.haversine` | ok | 2.18 ms / 2.24 ms | 2.69 ms | 0.00 MB (hbm) | ✓ 0.0011×tol | 23.35x vs nitrix-jax |
| spherical_geodesic_distance | jax-cpu | n=512 | `cupy.spherical_geodesic_distance` | skipped | — | — | — | — | — |
| spherical_geodesic_distance | jax-cpu | n=512 | `nitrix-jax` | ok | 3.06 ms / 3.79 ms | 90.20 ms | 768 MB (rss) | ✓ 0.00024×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cpu | n=512 | `sklearn.haversine` | ok | 8.56 ms / 9.10 ms | 10.05 ms | 768 MB (rss) | ✓ 0.0013×tol | 2.79x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=512 | `cupy.spherical_geodesic_distance` | ok | 1.05 ms / 1.06 ms | 146.06 ms | 0.01 MB (hbm) | ✓ 0.00021×tol | 11.44x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=512 | `nitrix-jax` | ok | 91.9 µs / 92.6 µs | 319.56 ms | 36.70 MB (hbm) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| spherical_geodesic_distance | jax-cuda12 | n=512 | `sklearn.haversine` | ok | 9.35 ms / 11.19 ms | 9.04 ms | 0.01 MB (hbm) | ✓ 0.0013×tol | 101.69x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

