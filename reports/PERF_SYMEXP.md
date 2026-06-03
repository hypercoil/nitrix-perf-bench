# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 645ce27d898f29997eff5632fb251170ec24d312 | bench: ac82ab53f1b7f1fb1d49e9d24b74435739dfaeec
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T02:37:20.597499+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| symexp | jax-cpu | d=256 | `cupy.eigh_expm` | skipped | — | — | — | — | — |
| symexp | jax-cpu | d=256 | `nitrix-jax` | ok | 5.06 ms / 13.49 ms | 435.89 ms | 715 MB (rss) | ✓ 0.0082×tol | 1.00x vs nitrix-jax |
| symexp | jax-cpu | d=256 | `scipy.linalg.expm` | ok | 66.77 ms / 188.55 ms | 148.42 ms | 715 MB (rss) | ✓ 0.003×tol | 13.21x vs nitrix-jax |
| symexp | jax-cuda12 | d=256 | `cupy.eigh_expm` | skipped | — | — | — | — | — |
| symexp | jax-cuda12 | d=256 | `nitrix-jax` | ok | 1.97 ms / 2.02 ms | 701.85 ms | 72.09 MB (hbm) | ✓ 0.011×tol | 1.00x vs nitrix-jax |
| symexp | jax-cuda12 | d=256 | `scipy.linalg.expm` | ok | 160.64 ms / 301.31 ms | 54.06 ms | 0.26 MB (hbm) | ✓ 0.003×tol | 81.73x vs nitrix-jax |
| symexp | jax-cpu | d=512 | `cupy.eigh_expm` | skipped | — | — | — | — | — |
| symexp | jax-cpu | d=512 | `nitrix-jax` | ok | 24.85 ms / 36.70 ms | 954.85 ms | 715 MB (rss) | ✓ 0.0068×tol | 1.00x vs nitrix-jax |
| symexp | jax-cpu | d=512 | `scipy.linalg.expm` | ok | 90.20 ms / 181.16 ms | 131.18 ms | 715 MB (rss) | ✓ 0.0029×tol | 3.63x vs nitrix-jax |
| symexp | jax-cuda12 | d=512 | `cupy.eigh_expm` | skipped | — | — | — | — | — |
| symexp | jax-cuda12 | d=512 | `nitrix-jax` | ok | 4.31 ms / 4.33 ms | 1.171 s | 74.45 MB (hbm) | ✓ 0.006×tol | 1.00x vs nitrix-jax |
| symexp | jax-cuda12 | d=512 | `scipy.linalg.expm` | ok | 59.96 ms / 130.71 ms | 177.53 ms | 1.05 MB (hbm) | ✓ 0.0029×tol | 13.92x vs nitrix-jax |
| symexp | jax-cpu | d=64 | `cupy.eigh_expm` | skipped | — | — | — | — | — |
| symexp | jax-cpu | d=64 | `nitrix-jax` | ok | 311.8 µs / 340.9 µs | 157.41 ms | 715 MB (rss) | ✓ 0.0082×tol | 1.00x vs nitrix-jax |
| symexp | jax-cpu | d=64 | `scipy.linalg.expm` | ok | 658.4 µs / 1.19 ms | 16.48 ms | 715 MB (rss) | ✓ 0.0019×tol | 2.11x vs nitrix-jax |
| symexp | jax-cuda12 | d=64 | `cupy.eigh_expm` | ok | 703.0 µs / 711.9 µs | 218.23 ms | 0.02 MB (hbm) | ✓ 0.01×tol | 0.94x vs nitrix-jax |
| symexp | jax-cuda12 | d=64 | `nitrix-jax` | ok | 745.4 µs / 755.7 µs | 475.69 ms | 71.35 MB (hbm) | ✓ 0.0099×tol | 1.00x vs nitrix-jax |
| symexp | jax-cuda12 | d=64 | `scipy.linalg.expm` | ok | 16.10 ms / 19.99 ms | 13.73 ms | 0.02 MB (hbm) | ✓ 0.0019×tol | 21.61x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

