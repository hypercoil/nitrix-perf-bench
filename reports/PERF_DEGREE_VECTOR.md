# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 88a23053af93d9466c3993dae0309eddd5c11c6f | bench: f174a26b2600f58f1ad06be86aeda31329d3d37c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T05:41:16.464105+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| degree_vector | jax-cpu | n=128 | `cupy.degree` | skipped | — | — | — | — | — |
| degree_vector | jax-cpu | n=128 | `nitrix-jax` | ok | 11.7 µs / 12.3 µs | 69.43 ms | 702 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cpu | n=128 | `numpy.degree` | ok | 6.6 µs / 6.6 µs | 17.7 µs | 702 MB (rss) | ✓ 0.00011×tol | 0.56x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=128 | `cupy.degree` | ok | 16.9 µs / 18.5 µs | 21.53 ms | 0.07 MB (hbm) | ✓ 0.00012×tol | 0.18x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=128 | `nitrix-jax` | ok | 95.8 µs / 97.7 µs | 224.73 ms | 33.62 MB (hbm) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=128 | `numpy.degree` | ok | 8.0 µs / 8.4 µs | 20.7 µs | 0.07 MB (hbm) | ✓ 0.00011×tol | 0.08x vs nitrix-jax |
| degree_vector | jax-cpu | n=256 | `cupy.degree` | skipped | — | — | — | — | — |
| degree_vector | jax-cpu | n=256 | `nitrix-jax` | ok | 12.5 µs / 17.0 µs | 39.31 ms | 702 MB (rss) | ✓ 0.00011×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cpu | n=256 | `numpy.degree` | ok | 17.0 µs / 17.3 µs | 27.1 µs | 702 MB (rss) | ✓ 0.00011×tol | 1.35x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=256 | `cupy.degree` | ok | 16.2 µs / 17.9 µs | 21.74 ms | 0.26 MB (hbm) | ✓ 0.00013×tol | 0.18x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=256 | `nitrix-jax` | ok | 91.2 µs / 95.4 µs | 192.95 ms | 33.82 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=256 | `numpy.degree` | ok | 21.8 µs / 26.8 µs | 37.5 µs | 0.26 MB (hbm) | ✓ 0.00011×tol | 0.24x vs nitrix-jax |
| degree_vector | jax-cpu | n=512 | `cupy.degree` | skipped | — | — | — | — | — |
| degree_vector | jax-cpu | n=512 | `nitrix-jax` | ok | 41.6 µs / 45.0 µs | 37.52 ms | 702 MB (rss) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cpu | n=512 | `numpy.degree` | ok | 56.4 µs / 57.1 µs | 67.5 µs | 702 MB (rss) | ✓ 0.00013×tol | 1.36x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=512 | `cupy.degree` | ok | 17.2 µs / 19.3 µs | 21.52 ms | 1.05 MB (hbm) | ✓ 0.00014×tol | 0.19x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=512 | `nitrix-jax` | ok | 91.2 µs / 95.5 µs | 190.87 ms | 34.61 MB (hbm) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| degree_vector | jax-cuda12 | n=512 | `numpy.degree` | ok | 56.8 µs / 57.4 µs | 77.1 µs | 1.05 MB (hbm) | ✓ 0.00013×tol | 0.62x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

