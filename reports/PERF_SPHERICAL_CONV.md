# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f42e7ff8398f69ecf54856f951b670c47199333b | bench: ddc528b51702f8a1aa44e81fec1f5da51b05fd09
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-04T06:10:02.399427+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| spherical_conv | jax-cpu | n=1024 | `cupy.spherical_conv` | skipped | — | — | — | — | — |
| spherical_conv | jax-cpu | n=1024 | `nitrix-jax` | ok | 16.56 ms / 19.04 ms | 326.23 ms | 767 MB (rss) | ✓ 0.00073×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cpu | n=1024 | `numpy.spherical_conv` | ok | 87.48 ms / 96.39 ms | 105.40 ms | 767 MB (rss) | ✓ 0.0011×tol | 5.28x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=1024 | `cupy.spherical_conv` | ok | 4.48 ms / 4.55 ms | 180.06 ms | 0.08 MB (hbm) | ✓ 0.00082×tol | 12.17x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=1024 | `nitrix-jax` | ok | 367.8 µs / 382.3 µs | 665.98 ms | 84.16 MB (hbm) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=1024 | `numpy.spherical_conv` | ok | 84.84 ms / 86.07 ms | 86.31 ms | 0.08 MB (hbm) | ✓ 0.0011×tol | 230.68x vs nitrix-jax |
| spherical_conv | jax-cpu | n=256 | `cupy.spherical_conv` | skipped | — | — | — | — | — |
| spherical_conv | jax-cpu | n=256 | `nitrix-jax` | ok | 1.25 ms / 1.32 ms | 272.08 ms | 767 MB (rss) | ✓ 0.0011×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cpu | n=256 | `numpy.spherical_conv` | ok | 5.90 ms / 5.96 ms | 6.12 ms | 767 MB (rss) | ✓ 0.001×tol | 4.73x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=256 | `cupy.spherical_conv` | ok | 950.8 µs / 987.3 µs | 16.855 s | 0.02 MB (hbm) | ✓ 0.0015×tol | 5.50x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=256 | `nitrix-jax` | ok | 172.7 µs / 190.1 µs | 695.99 ms | 67.16 MB (hbm) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=256 | `numpy.spherical_conv` | ok | 7.87 ms / 8.01 ms | 7.42 ms | 0.02 MB (hbm) | ✓ 0.001×tol | 45.57x vs nitrix-jax |
| spherical_conv | jax-cpu | n=512 | `cupy.spherical_conv` | skipped | — | — | — | — | — |
| spherical_conv | jax-cpu | n=512 | `nitrix-jax` | ok | 4.85 ms / 5.58 ms | 366.02 ms | 767 MB (rss) | ✓ 0.001×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cpu | n=512 | `numpy.spherical_conv` | ok | 21.73 ms / 25.47 ms | 23.61 ms | 767 MB (rss) | ✓ 0.00079×tol | 4.48x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=512 | `cupy.spherical_conv` | ok | 1.67 ms / 1.69 ms | 192.38 ms | 0.04 MB (hbm) | ✓ 0.001×tol | 8.49x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=512 | `nitrix-jax` | ok | 196.5 µs / 200.2 µs | 896.86 ms | 85.11 MB (hbm) | ✓ 0.00092×tol | 1.00x vs nitrix-jax |
| spherical_conv | jax-cuda12 | n=512 | `numpy.spherical_conv` | ok | 21.47 ms / 26.02 ms | 29.21 ms | 0.04 MB (hbm) | ✓ 0.00079×tol | 109.29x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

