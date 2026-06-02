# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: d0a9ca5fc20f2136415cfd5d76f4257fba31857a | bench: a1af688c287be74a0019e7ceb96677e6ab023820
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T23:13:36.117259+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| precision | jax-cpu | c=128,obs=1024 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=128,obs=1024 | `nilearn.precision` | ok | 96.18 ms / 184.12 ms | 1.666 s | 754 MB (rss) | ✓ 0.91×tol | 135.46x vs nitrix-jax |
| precision | jax-cpu | c=128,obs=1024 | `nitrix-jax` | ok | 710.0 µs / 858.0 µs | 235.96 ms | 754 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=128,obs=1024 | `numpy.inv_cov` | ok | 748.1 µs / 770.2 µs | 802.1 µs | 754 MB (rss) | ✓ 9.9e-05×tol | 1.05x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `cupy.inv_cov` | ok | 714.6 µs / 732.3 µs | 1.148 s | 0.52 MB (hbm) | ✓ 0.00072×tol | 2.02x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `nilearn.precision` | ok | 109.02 ms / 182.05 ms | 944.10 ms | 0.52 MB (hbm) | ✓ 0.91×tol | 307.61x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `nitrix-jax` | ok | 354.4 µs / 363.4 µs | 608.09 ms | 72.88 MB (hbm) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `numpy.inv_cov` | ok | 757.6 µs / 772.9 µs | 483.13 ms | 0.52 MB (hbm) | ✓ 9.9e-05×tol | 2.14x vs nitrix-jax |
| precision | jax-cpu | c=256,obs=2048 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=256,obs=2048 | `nilearn.precision` | ok | 172.35 ms / 293.76 ms | 1.086 s | 754 MB (rss) | ✓ 0.45×tol | 15.00x vs nitrix-jax |
| precision | jax-cpu | c=256,obs=2048 | `nitrix-jax` | ok | 11.49 ms / 40.94 ms | 496.70 ms | 754 MB (rss) | ✓ 0.0017×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=256,obs=2048 | `numpy.inv_cov` | ok | 4.09 ms / 30.30 ms | 59.26 ms | 754 MB (rss) | ✓ 0.0001×tol | 0.36x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `cupy.inv_cov` | ok | 1.37 ms / 1.40 ms | 231.96 ms | 2.10 MB (hbm) | ✓ 0.00077×tol | 2.23x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `nilearn.precision` | ok | 123.27 ms / 197.66 ms | 1.516 s | 2.10 MB (hbm) | ✓ 0.45×tol | 200.86x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `nitrix-jax` | ok | 613.7 µs / 632.6 µs | 598.71 ms | 77.59 MB (hbm) | ✓ 0.00091×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `numpy.inv_cov` | ok | 3.85 ms / 4.88 ms | 3.94 ms | 2.10 MB (hbm) | ✓ 0.0001×tol | 6.28x vs nitrix-jax |
| precision | jax-cpu | c=512,obs=4096 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=512,obs=4096 | `nilearn.precision` | ok | 476.51 ms / 578.92 ms | 1.674 s | 754 MB (rss) | ✓ 0.23×tol | 17.82x vs nitrix-jax |
| precision | jax-cpu | c=512,obs=4096 | `nitrix-jax` | ok | 26.75 ms / 32.18 ms | 362.89 ms | 754 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=512,obs=4096 | `numpy.inv_cov` | ok | 26.25 ms / 35.01 ms | 44.74 ms | 754 MB (rss) | ✓ 0.00011×tol | 0.98x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `cupy.inv_cov` | ok | 6.79 ms / 6.82 ms | 323.32 ms | 8.39 MB (hbm) | ✓ 0.0011×tol | 4.25x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `nilearn.precision` | ok | 381.26 ms / 486.72 ms | 1.131 s | 8.39 MB (hbm) | ✓ 0.23×tol | 238.51x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `nitrix-jax` | ok | 1.60 ms / 1.61 ms | 573.86 ms | 88.08 MB (hbm) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `numpy.inv_cov` | ok | 190.09 ms / 1.434 s | 1.396 s | 8.39 MB (hbm) | ✓ 0.00011×tol | 118.92x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

