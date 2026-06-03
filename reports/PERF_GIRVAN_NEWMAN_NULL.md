# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 88a23053af93d9466c3993dae0309eddd5c11c6f | bench: f174a26b2600f58f1ad06be86aeda31329d3d37c
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T05:42:02.787062+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| girvan_newman_null | jax-cpu | n=128 | `cupy.gn_null` | skipped | — | — | — | — | — |
| girvan_newman_null | jax-cpu | n=128 | `nitrix-jax` | ok | 30.2 µs / 35.1 µs | 109.09 ms | 705 MB (rss) | ✓ 0.00014×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cpu | n=128 | `numpy.gn_null` | ok | 22.1 µs / 22.6 µs | 56.7 µs | 705 MB (rss) | ✓ 0.00013×tol | 0.73x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=128 | `cupy.gn_null` | ok | 115.0 µs / 121.4 µs | 286.27 ms | 0.07 MB (hbm) | ✓ 0.00011×tol | 1.14x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=128 | `nitrix-jax` | ok | 100.5 µs / 103.6 µs | 348.03 ms | 33.62 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=128 | `numpy.gn_null` | ok | 32.2 µs / 32.9 µs | 73.7 µs | 0.07 MB (hbm) | ✓ 0.00013×tol | 0.32x vs nitrix-jax |
| girvan_newman_null | jax-cpu | n=256 | `cupy.gn_null` | skipped | — | — | — | — | — |
| girvan_newman_null | jax-cpu | n=256 | `nitrix-jax` | ok | 52.9 µs / 63.2 µs | 79.72 ms | 705 MB (rss) | ✓ 0.00019×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cpu | n=256 | `numpy.gn_null` | ok | 67.3 µs / 68.5 µs | 96.5 µs | 705 MB (rss) | ✓ 0.00017×tol | 1.27x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=256 | `cupy.gn_null` | ok | 116.0 µs / 118.8 µs | 157.03 ms | 0.26 MB (hbm) | ✓ 0.00018×tol | 1.16x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=256 | `nitrix-jax` | ok | 99.8 µs / 101.6 µs | 366.58 ms | 33.82 MB (hbm) | ✓ 0.00012×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=256 | `numpy.gn_null` | ok | 65.1 µs / 66.9 µs | 95.0 µs | 0.26 MB (hbm) | ✓ 0.00017×tol | 0.65x vs nitrix-jax |
| girvan_newman_null | jax-cpu | n=512 | `cupy.gn_null` | skipped | — | — | — | — | — |
| girvan_newman_null | jax-cpu | n=512 | `nitrix-jax` | ok | 141.1 µs / 159.7 µs | 86.76 ms | 705 MB (rss) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cpu | n=512 | `numpy.gn_null` | ok | 223.9 µs / 243.1 µs | 333.8 µs | 705 MB (rss) | ✓ 0.00013×tol | 1.59x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=512 | `cupy.gn_null` | ok | 116.1 µs / 119.9 µs | 166.68 ms | 1.05 MB (hbm) | ✓ 0.00015×tol | 1.15x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=512 | `nitrix-jax` | ok | 100.9 µs / 105.5 µs | 336.23 ms | 34.61 MB (hbm) | ✓ 0.00013×tol | 1.00x vs nitrix-jax |
| girvan_newman_null | jax-cuda12 | n=512 | `numpy.gn_null` | ok | 216.4 µs / 220.2 µs | 242.3 µs | 1.05 MB (hbm) | ✓ 0.00013×tol | 2.14x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

