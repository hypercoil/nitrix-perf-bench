# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 6f669403bec6091b1db461932fd6a10dda3b6a87 | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-10T23:38:02.000475+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | ok | 5.658 s / 5.732 s | 10.963 s | 828 MB (rss) | n/a (no oracle) | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | timeout | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[128, 128, 128],levels=2,iters=80 | `nitrix-jax` | ok | 2.460 s / 2.493 s | 39.364 s | 8808.57 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | ok | 450.32 ms / 451.78 ms | 7.244 s | 816 MB (rss) | n/a (no oracle) | 0.18x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 2.489 s / 2.531 s | 4.416 s | 816 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 32.72 ms / 33.01 ms | 5.487 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | ok | 463.15 ms / 467.22 ms | 3.833 s | 816 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 5.855 s / 5.881 s | 11.279 s | 816 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 122.55 ms / 152.95 ms | 12.629 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | ok | 474.61 ms / 479.13 ms | 3.671 s | 816 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 5.648 s / 5.741 s | 13.331 s | 816 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 197.69 ms / 242.42 ms | 18.653 s | 339.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | ok | 887.23 ms / 905.62 ms | 4.845 s | 816 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 25.060 s / 25.467 s | 31.593 s | 816 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80 | `nitrix-jax` | ok | 171.35 ms / 206.94 ms | 13.990 s | 1711.41 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 1.892 s / 1.892 s | 6.316 s | 816 MB (rss) | n/a (no oracle) | 0.07x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 26.408 s / 26.550 s | 33.906 s | 849 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[64, 64, 64],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 195.74 ms / 206.61 ms | 16.336 s | 1711.41 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | ok | 2.783 s / 2.840 s | 9.847 s | 816 MB (rss) | n/a (no oracle) | 0.03x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | ok | 80.798 s / 81.045 s | 91.822 s | 986 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80 | `nitrix-jax` | ok | 742.59 ms / 749.97 ms | 17.741 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | ok | 5.936 s / 6.230 s | 14.206 s | 816 MB (rss) | n/a (no oracle) | 0.07x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cpu | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 81.030 s / 82.644 s | 105.707 s | 976 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[96, 96, 96],levels=2,iters=80,spacing=[1, 1, 3] | `nitrix-jax` | ok | 768.63 ms / 784.58 ms | 21.022 s | 3777.37 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

