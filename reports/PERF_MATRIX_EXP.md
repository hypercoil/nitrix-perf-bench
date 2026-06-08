# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 3938db4e71ef270b133e0f138b65a816703d111e | bench: 6898ed0318d6340ff65112010b9b86d96eb1e522
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-08T22:10:21.613331+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| matrix_exp | jax-cuda12 | n=1024 | `jax.scipy.linalg.expm` | ok | 4.76 ms / 4.77 ms | 709.21 ms | 83.89 MB (hbm) | ✓ 0.01×tol | 1.39x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=1024 | `nitrix-jax` | ok | 3.42 ms / 3.54 ms | 316.73 ms | 104.86 MB (hbm) | ✓ 0.17×tol | 1.00x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=1024 | `scipy.linalg.expm` | ok | 89.16 ms / 93.14 ms | 169.65 ms | 4.19 MB (hbm) | ✓ 0.00092×tol | 26.07x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=16 | `jax.scipy.linalg.expm` | ok | 497.5 µs / 503.4 µs | 1.211 s | 71.31 MB (hbm) | ✓ 0.00075×tol | 3.14x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=16 | `nitrix-jax` | ok | 158.4 µs / 170.8 µs | 429.58 ms | 88.08 MB (hbm) | ✓ 0.029×tol | 1.00x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=16 | `scipy.linalg.expm` | ok | 25.1 µs / 27.8 µs | 105.4 µs | 0.00 MB (hbm) | ✓ 0.00099×tol | 0.16x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=256 | `jax.scipy.linalg.expm` | ok | 934.5 µs / 960.1 µs | 678.28 ms | 72.09 MB (hbm) | ✓ 0.0036×tol | 3.43x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=256 | `nitrix-jax` | ok | 272.2 µs / 273.7 µs | 325.46 ms | 89.13 MB (hbm) | ✓ 0.061×tol | 1.00x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=256 | `scipy.linalg.expm` | ok | 19.97 ms / 90.08 ms | 157.16 ms | 0.26 MB (hbm) | ✓ 0.0019×tol | 73.38x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=512 | `jax.scipy.linalg.expm` | ok | 1.91 ms / 1.92 ms | 836.51 ms | 74.45 MB (hbm) | ✓ 0.011×tol | 2.84x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=512 | `nitrix-jax` | ok | 671.6 µs / 684.5 µs | 319.46 ms | 92.27 MB (hbm) | ✓ 0.1×tol | 1.00x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=512 | `scipy.linalg.expm` | ok | 14.37 ms / 15.28 ms | 92.53 ms | 1.05 MB (hbm) | ✓ 0.001×tol | 21.40x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=64 | `jax.scipy.linalg.expm` | ok | 579.6 µs / 673.9 µs | 773.50 ms | 71.35 MB (hbm) | ✓ 0.002×tol | 3.09x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=64 | `nitrix-jax` | ok | 187.3 µs / 189.3 µs | 303.06 ms | 88.15 MB (hbm) | ✓ 0.049×tol | 1.00x vs nitrix-jax |
| matrix_exp | jax-cuda12 | n=64 | `scipy.linalg.expm` | ok | 169.4 µs / 177.9 µs | 12.59 ms | 0.02 MB (hbm) | ✓ 0.00079×tol | 0.90x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

