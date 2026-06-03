# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 32fd5ab9420d25d8be13008bc3b162856e0fcad7 | bench: 8157e5089aaf5c007d82fc83adc5ffc5a9c2874c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T21:27:01.000319+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| coaffiliation | jax-cpu | n=128 | `cupy.coaffiliation` | skipped | — | — | — | — | — |
| coaffiliation | jax-cpu | n=128 | `nitrix-jax` | ok | 32.4 µs / 34.7 µs | 70.33 ms | 706 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cpu | n=128 | `numpy.coaffiliation` | ok | 27.4 µs / 29.0 µs | 76.9 µs | 706 MB (rss) | ✓ 0.00025×tol | 0.85x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=128 | `cupy.coaffiliation` | ok | 113.1 µs / 116.5 µs | 126.01 ms | 0.01 MB (hbm) | ✓ 0.00017×tol | 1.08x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=128 | `nitrix-jax` | ok | 104.7 µs / 106.8 µs | 209.49 ms | 71.44 MB (hbm) | ✓ 0.00016×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=128 | `numpy.coaffiliation` | ok | 30.4 µs / 31.0 µs | 77.8 µs | 0.01 MB (hbm) | ✓ 0.00025×tol | 0.29x vs nitrix-jax |
| coaffiliation | jax-cpu | n=256 | `cupy.coaffiliation` | skipped | — | — | — | — | — |
| coaffiliation | jax-cpu | n=256 | `nitrix-jax` | ok | 93.8 µs / 98.5 µs | 98.72 ms | 706 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cpu | n=256 | `numpy.coaffiliation` | ok | 132.2 µs / 132.8 µs | 206.6 µs | 706 MB (rss) | ✓ 0.00025×tol | 1.41x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=256 | `cupy.coaffiliation` | ok | 112.0 µs / 116.2 µs | 121.48 ms | 0.02 MB (hbm) | ✓ 0.00018×tol | 1.02x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=256 | `nitrix-jax` | ok | 109.7 µs / 117.7 µs | 251.45 ms | 71.84 MB (hbm) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=256 | `numpy.coaffiliation` | ok | 132.3 µs / 134.1 µs | 203.8 µs | 0.02 MB (hbm) | ✓ 0.00025×tol | 1.21x vs nitrix-jax |
| coaffiliation | jax-cpu | n=512 | `cupy.coaffiliation` | skipped | — | — | — | — | — |
| coaffiliation | jax-cpu | n=512 | `nitrix-jax` | ok | 222.7 µs / 297.0 µs | 109.96 ms | 706 MB (rss) | ✓ 0.00028×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cpu | n=512 | `numpy.coaffiliation` | ok | 474.0 µs / 475.4 µs | 1.65 ms | 706 MB (rss) | ✓ 0.00028×tol | 2.13x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=512 | `cupy.coaffiliation` | ok | 113.7 µs / 120.0 µs | 108.84 ms | 0.03 MB (hbm) | ✓ 0.00021×tol | 1.06x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=512 | `nitrix-jax` | ok | 107.3 µs / 110.6 µs | 239.80 ms | 74.45 MB (hbm) | ✓ 0.00028×tol | 1.00x vs nitrix-jax |
| coaffiliation | jax-cuda12 | n=512 | `numpy.coaffiliation` | ok | 468.7 µs / 471.2 µs | 600.6 µs | 0.03 MB (hbm) | ✓ 0.00028×tol | 4.37x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

