# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 9d1cdef349e8e7411de8ab0275bd07b604ec81ed | bench: 4f11ef0ed14d01b428d039a9e0fc26e395d7b120
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-12T03:21:54.981427+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| flame_two_level | jax-cpu | V=1024,N=60 | `fsl.flameo` | ok | 732.81 ms / 756.19 ms | 991.81 ms | 1298 MB (rss) | ✓ 0.23×tol | — |
| flame_two_level | jax-cpu | V=1024,N=60 | `fsl.iofloor` | ok | 234.63 ms / 243.75 ms | 489.18 ms | 1298 MB (rss) | ≈ 1.7e+02×tol | — |
| flame_two_level | jax-cpu | V=1024,N=60 | `nitrix-jax` | ok | 104.77 ms / 108.13 ms | 1.278 s | 1299 MB (rss) | ✓ 0.029×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cpu | V=1024,N=60 | `statsmodels.meta_analysis` | ok | 308.02 ms / 309.45 ms | 2.579 s | 1299 MB (rss) | ✓ 2.4e-05×tol | 2.94x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=1024,N=60 | `fsl.flameo` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=1024,N=60 | `fsl.iofloor` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=1024,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=1024,N=60 | `statsmodels.meta_analysis` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=131072,N=60 | `fsl.flameo` | ok | 31.329 s / 38.101 s | 36.744 s | 1298 MB (rss) | ≈ 2.2×tol | — |
| flame_two_level | jax-cpu | V=131072,N=60 | `fsl.iofloor` | ok | 3.651 s / 3.815 s | 4.401 s | 1298 MB (rss) | ≈ 1.7e+02×tol | — |
| flame_two_level | jax-cpu | V=131072,N=60 | `nitrix-jax` | ok | 20.267 s / 20.602 s | 22.225 s | 1299 MB (rss) | ✓ 0.035×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cpu | V=131072,N=60 | `statsmodels.meta_analysis` | ok | 39.238 s / 39.710 s | 42.341 s | 1299 MB (rss) | ✓ 2.4e-05×tol | 1.94x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=131072,N=60 | `fsl.flameo` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=131072,N=60 | `fsl.iofloor` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=131072,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=131072,N=60 | `statsmodels.meta_analysis` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=262144,N=60 | `fsl.flameo` | ok | 63.134 s / 66.176 s | 66.883 s | 1298 MB (rss) | ≈ 2.2×tol | — |
| flame_two_level | jax-cpu | V=262144,N=60 | `fsl.iofloor` | ok | 6.455 s / 6.602 s | 7.167 s | 1298 MB (rss) | ≈ 1.7e+02×tol | — |
| flame_two_level | jax-cpu | V=262144,N=60 | `nitrix-jax` | timeout | — | — | — | — | — |
| flame_two_level | jax-cpu | V=262144,N=60 | `statsmodels.meta_analysis` | ok | 80.246 s / 80.406 s | 82.399 s | 1299 MB (rss) | ✓ 2.4e-05×tol | — |
| flame_two_level | jax-cuda12 | V=262144,N=60 | `fsl.flameo` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=262144,N=60 | `fsl.iofloor` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=262144,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=262144,N=60 | `statsmodels.meta_analysis` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=65536,N=60 | `fsl.flameo` | ok | 15.434 s / 15.632 s | 15.832 s | 1298 MB (rss) | ✓ 0.51×tol | — |
| flame_two_level | jax-cpu | V=65536,N=60 | `fsl.iofloor` | ok | 1.764 s / 1.818 s | 1.965 s | 1298 MB (rss) | ≈ 1.7e+02×tol | — |
| flame_two_level | jax-cpu | V=65536,N=60 | `nitrix-jax` | ok | 9.610 s / 9.659 s | 11.154 s | 1299 MB (rss) | ✓ 0.025×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cpu | V=65536,N=60 | `statsmodels.meta_analysis` | ok | 19.466 s / 19.725 s | 21.798 s | 1299 MB (rss) | ✓ 2.4e-05×tol | 2.03x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=65536,N=60 | `fsl.flameo` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=65536,N=60 | `fsl.iofloor` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=65536,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=65536,N=60 | `statsmodels.meta_analysis` | skipped | — | — | — | — | — |
| flame_two_level | jax-cpu | V=8192,N=60 | `fsl.flameo` | ok | 2.353 s / 2.387 s | 2.548 s | 1298 MB (rss) | ✓ 0.51×tol | — |
| flame_two_level | jax-cpu | V=8192,N=60 | `fsl.iofloor` | ok | 414.14 ms / 421.48 ms | 547.45 ms | 1298 MB (rss) | ≈ 1.7e+02×tol | — |
| flame_two_level | jax-cpu | V=8192,N=60 | `nitrix-jax` | ok | 1.017 s / 1.033 s | 2.596 s | 1299 MB (rss) | ✓ 0.019×tol | 1.00x vs nitrix-jax |
| flame_two_level | jax-cpu | V=8192,N=60 | `statsmodels.meta_analysis` | ok | 2.372 s / 2.396 s | 4.300 s | 1299 MB (rss) | ✓ 2.4e-05×tol | 2.33x vs nitrix-jax |
| flame_two_level | jax-cuda12 | V=8192,N=60 | `fsl.flameo` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=8192,N=60 | `fsl.iofloor` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=8192,N=60 | `nitrix-jax` | skipped | — | — | — | — | — |
| flame_two_level | jax-cuda12 | V=8192,N=60 | `statsmodels.meta_analysis` | skipped | — | — | — | — | — |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

