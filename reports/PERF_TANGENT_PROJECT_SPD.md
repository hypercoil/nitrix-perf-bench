# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: f04ec5092b2bbb8ecd3b17b7f4f268470999f4d9 | bench: f8a8b86b0d31bd387352b0affc529a97cd9a46cb
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T20:08:37.181901+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| tangent_project_spd | jax-cpu | b=32,d=256 | `cupy.eigh_tangent` | skipped | — | — | — | — | — |
| tangent_project_spd | jax-cpu | b=32,d=256 | `nilearn.tangent` | ok | 241.97 ms / 246.21 ms | 1.177 s | 768 MB (rss) | ✓ 0.0042×tol | 1.09x vs nitrix-jax |
| tangent_project_spd | jax-cpu | b=32,d=256 | `nitrix-jax` | ok | 222.53 ms / 332.85 ms | 743.70 ms | 771 MB (rss) | ✓ 0.00097×tol | 1.00x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=32,d=256 | `cupy.eigh_tangent` | skipped | — | — | — | — | — |
| tangent_project_spd | jax-cuda12 | b=32,d=256 | `nilearn.tangent` | ok | 229.31 ms / 231.23 ms | 795.14 ms | 8.65 MB (hbm) | ✓ 0.0042×tol | 25.49x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=32,d=256 | `nitrix-jax` | ok | 9.00 ms / 9.01 ms | 1.048 s | 126.09 MB (hbm) | ✓ 0.0009×tol | 1.00x vs nitrix-jax |
| tangent_project_spd | jax-cpu | b=64,d=128 | `cupy.eigh_tangent` | skipped | — | — | — | — | — |
| tangent_project_spd | jax-cpu | b=64,d=128 | `nilearn.tangent` | ok | 108.22 ms / 130.10 ms | 730.79 ms | 768 MB (rss) | ✓ 0.0038×tol | 1.03x vs nitrix-jax |
| tangent_project_spd | jax-cpu | b=64,d=128 | `nitrix-jax` | ok | 104.83 ms / 154.46 ms | 393.66 ms | 768 MB (rss) | ✓ 0.00074×tol | 1.00x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=64,d=128 | `cupy.eigh_tangent` | compile_error | — | — | — | — | — |
| tangent_project_spd | jax-cuda12 | b=64,d=128 | `nilearn.tangent` | ok | 149.98 ms / 186.28 ms | 1.010 s | 4.26 MB (hbm) | ✓ 0.0038×tol | 32.08x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=64,d=128 | `nitrix-jax` | ok | 4.68 ms / 4.73 ms | 1.291 s | 104.92 MB (hbm) | ✓ 0.00063×tol | 1.00x vs nitrix-jax |
| tangent_project_spd | jax-cpu | b=64,d=64 | `cupy.eigh_tangent` | skipped | — | — | — | — | — |
| tangent_project_spd | jax-cpu | b=64,d=64 | `nilearn.tangent` | ok | 27.38 ms / 31.34 ms | 3.141 s | 768 MB (rss) | ✓ 0.0016×tol | 1.35x vs nitrix-jax |
| tangent_project_spd | jax-cpu | b=64,d=64 | `nitrix-jax` | ok | 20.35 ms / 25.18 ms | 340.98 ms | 768 MB (rss) | ✓ 0.0006×tol | 1.00x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=64,d=64 | `cupy.eigh_tangent` | compile_error | — | — | — | — | — |
| tangent_project_spd | jax-cuda12 | b=64,d=64 | `nilearn.tangent` | ok | 26.44 ms / 27.02 ms | 601.61 ms | 1.06 MB (hbm) | ✓ 0.0016×tol | 15.04x vs nitrix-jax |
| tangent_project_spd | jax-cuda12 | b=64,d=64 | `nitrix-jax` | ok | 1.76 ms / 1.82 ms | 1.165 s | 74.47 MB (hbm) | ✓ 0.00064×tol | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

