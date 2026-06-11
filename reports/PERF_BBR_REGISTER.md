# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c54bc81807fd0b81c371b5904a98e8e6f3d88a93 | bench: 7be151160d256117f2a68003be1befe98a76a202
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-11T16:57:54.765205+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 3.72 ms / 3.75 ms | 3.232 s | 694 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 6.32 ms / 6.37 ms | 7.337 s | 336.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 3.65 ms / 3.75 ms | 3.715 s | 683 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 6.08 ms / 6.27 ms | 9.121 s | 336.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=20000,iters=100 | `nitrix-jax` | ok | 43.77 ms / 44.65 ms | 3.129 s | 700 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=20000,iters=100 | `nitrix-jax` | ok | 3.44 ms / 3.48 ms | 7.802 s | 337.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=5000,iters=100 | `nitrix-jax` | ok | 22.96 ms / 24.32 ms | 3.110 s | 697 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=5000,iters=100 | `nitrix-jax` | ok | 9.63 ms / 9.66 ms | 7.638 s | 336.72 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=80000,iters=100 | `nitrix-jax` | ok | 174.33 ms / 176.75 ms | 3.265 s | 731 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=80000,iters=100 | `nitrix-jax` | ok | 5.86 ms / 5.96 ms | 6.155 s | 338.61 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

