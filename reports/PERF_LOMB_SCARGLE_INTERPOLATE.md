# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: e0b2ff20bb0e32c3184030bd6df4a6e0d4898d74 | bench: 32d06bc31b97a6f8ae9be9a29128165a28860849
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T07:10:59.178045+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=1024 | `cupy.joint_glm` | skipped | — | — | — | — | — |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=1024 | `nitrix-jax` | ok | 152.69 ms / 186.50 ms | 2.616 s | 832 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=1024 | `numpy.joint_glm` | ok | 194.09 ms / 224.28 ms | 319.90 ms | 644 MB (rss) | n/a (no oracle) | 1.27x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=1024 | `cupy.joint_glm` | ok | 6.80 ms / 6.81 ms | 281.15 ms | 16.78 MB (hbm) | n/a (no oracle) | 1.07x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=1024 | `nitrix-jax` | ok | 6.33 ms / 7.77 ms | 1.304 s | 121.64 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=1024 | `numpy.joint_glm` | ok | 136.27 ms / 140.36 ms | 171.89 ms | 16.78 MB (hbm) | n/a (no oracle) | 21.52x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=256 | `cupy.joint_glm` | skipped | — | — | — | — | — |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=256 | `nitrix-jax` | ok | 18.87 ms / 22.83 ms | 214.87 ms | 659 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=256 | `numpy.joint_glm` | ok | 12.50 ms / 18.36 ms | 18.14 ms | 619 MB (rss) | n/a (no oracle) | 0.66x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=256 | `cupy.joint_glm` | ok | 1.44 ms / 1.46 ms | 2.921 s | 4.19 MB (hbm) | n/a (no oracle) | 1.22x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=256 | `nitrix-jax` | ok | 1.18 ms / 1.18 ms | 553.76 ms | 83.89 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=256 | `numpy.joint_glm` | ok | 11.90 ms / 167.25 ms | 717.80 ms | 4.19 MB (hbm) | n/a (no oracle) | 10.09x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=512 | `cupy.joint_glm` | skipped | — | — | — | — | — |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=512 | `nitrix-jax` | ok | 67.97 ms / 130.94 ms | 247.30 ms | 722 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cpu | V=4096,obs=512 | `numpy.joint_glm` | ok | 49.26 ms / 56.71 ms | 61.13 ms | 619 MB (rss) | n/a (no oracle) | 0.72x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=512 | `cupy.joint_glm` | ok | 2.86 ms / 2.88 ms | 240.90 ms | 8.39 MB (hbm) | n/a (no oracle) | 1.10x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=512 | `nitrix-jax` | ok | 2.60 ms / 2.62 ms | 705.71 ms | 100.66 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| lomb_scargle_interpolate | jax-cuda12 | V=4096,obs=512 | `numpy.joint_glm` | ok | 94.92 ms / 2.315 s | 1.667 s | 8.39 MB (hbm) | n/a (no oracle) | 36.46x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

