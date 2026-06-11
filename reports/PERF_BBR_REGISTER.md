# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 6f669403bec6091b1db461932fd6a10dda3b6a87 | bench: 602caa71af2b7fd2eeb43cd242362f4d81ece0c3
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-10T23:35:49.759188+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 3.66 ms / 3.72 ms | 3.424 s | 695 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 5.94 ms / 5.97 ms | 7.810 s | 336.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 3.62 ms / 3.70 ms | 3.189 s | 699 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 6.09 ms / 6.32 ms | 8.657 s | 336.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=20000,iters=100 | `nitrix-jax` | ok | 45.65 ms / 47.76 ms | 3.044 s | 702 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=20000,iters=100 | `nitrix-jax` | ok | 6.86 ms / 6.91 ms | 9.213 s | 337.08 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=5000,iters=100 | `nitrix-jax` | ok | 22.58 ms / 23.39 ms | 3.631 s | 701 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=5000,iters=100 | `nitrix-jax` | ok | 1.35 ms / 1.36 ms | 7.958 s | 336.72 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[64, 64, 64],N=80000,iters=100 | `nitrix-jax` | ok | 172.04 ms / 183.30 ms | 3.728 s | 746 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[64, 64, 64],N=80000,iters=100 | `nitrix-jax` | ok | 17.22 ms / 17.29 ms | 6.381 s | 338.61 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

