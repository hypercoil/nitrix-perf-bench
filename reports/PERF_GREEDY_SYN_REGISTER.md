# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c54bc81807fd0b81c371b5904a98e8e6f3d88a93 | bench: 7be151160d256117f2a68003be1befe98a76a202
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T20:02:47.517519+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `ants.registration` | ok | 4.507 s / 4.654 s | 9.709 s | 817 MB (rss) | n/a (no oracle) | 0.07x vs nitrix-jax |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `dipy.registration` | ok | 147.761 s / 148.447 s | 149.896 s | 817 MB (rss) | n/a (no oracle) | 2.41x vs nitrix-jax |
| greedy_syn_register | jax-cpu | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax` | ok | 61.400 s / 62.704 s | 71.951 s | 1103 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | data=mni152,resolution=2,levels=2,iters=80 | `nitrix-jax` | ok | 1.062 s / 1.064 s | 38.327 s | 4749.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | ok | 6.323 s / 6.341 s | 13.123 s | 830 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | ok | 109.522 s / 110.801 s | 112.192 s | 967 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | ok | 2.460 s / 2.462 s | 45.232 s | 8783.40 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | ok | 692.78 ms / 1.103 s | 6.325 s | 817 MB (rss) | n/a (no oracle) | 0.28x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | ok | 4.268 s / 4.282 s | 6.221 s | 817 MB (rss) | n/a (no oracle) | 1.73x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 2.464 s / 2.555 s | 4.191 s | 817 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 33.17 ms / 34.67 ms | 7.379 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | ok | 663.31 ms / 667.89 ms | 3.507 s | 817 MB (rss) | n/a (no oracle) | 0.11x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | ok | 10.495 s / 10.522 s | 11.143 s | 817 MB (rss) | n/a (no oracle) | 1.82x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 5.776 s / 6.062 s | 9.544 s | 817 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 122.58 ms / 124.05 ms | 18.573 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | ok | 652.29 ms / 667.43 ms | 4.227 s | 817 MB (rss) | n/a (no oracle) | 0.11x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | ok | 10.703 s / 10.899 s | 11.661 s | 817 MB (rss) | n/a (no oracle) | 1.81x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 5.905 s / 6.510 s | 11.907 s | 817 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 184.43 ms / 216.16 ms | 29.380 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | ok | 1.101 s / 1.101 s | 7.333 s | 817 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | ok | 13.077 s / 13.102 s | 14.582 s | 817 MB (rss) | n/a (no oracle) | 0.52x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 25.252 s / 26.001 s | 29.782 s | 835 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 178.89 ms / 208.53 ms | 21.967 s | 1711.41 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 2.109 s / 2.220 s | 8.941 s | 817 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | ok | 31.646 s / 31.647 s | 33.196 s | 817 MB (rss) | n/a (no oracle) | 1.20x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 26.350 s / 26.993 s | 31.678 s | 817 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 205.60 ms / 206.10 ms | 22.737 s | 1711.41 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | ok | 3.396 s / 3.563 s | 7.826 s | 817 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | ok | 103.194 s / 103.479 s | 104.495 s | 817 MB (rss) | n/a (no oracle) | 1.28x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | ok | 80.896 s / 81.200 s | 87.182 s | 970 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | ok | 741.84 ms / 782.51 ms | 21.648 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 6.430 s / 6.474 s | 10.933 s | 817 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | ok | 118.808 s / 119.076 s | 119.939 s | 817 MB (rss) | n/a (no oracle) | 1.44x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 82.542 s / 82.646 s | 88.860 s | 999 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 775.93 ms / 779.10 ms | 25.122 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

