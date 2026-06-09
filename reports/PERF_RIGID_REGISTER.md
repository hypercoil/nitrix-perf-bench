# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 38926ea2bfa0195dd61917fdc124b0a009e99382 | bench: 4d0cc033e0eb497c6344a4824a576b3c2410fdbf
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T07:45:38.723073+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 31.03 ms / 32.86 ms | 2.919 s | 703 MB (rss) | n/a (no oracle) | 0.11x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 272.65 ms / 281.09 ms | 5.277 s | 870 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 16.19 ms / 16.31 ms | 16.644 s | 437.10 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 31.21 ms / 33.85 ms | 1.765 s | 703 MB (rss) | n/a (no oracle) | 0.05x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 642.75 ms / 649.62 ms | 23.158 s | 1525 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 44.68 ms / 44.93 ms | 64.430 s | 437.10 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 32.61 ms / 36.53 ms | 2.387 s | 703 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 817.10 ms / 828.44 ms | 59.457 s | 2936 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 63.95 ms / 65.53 ms | 140.869 s | 437.10 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

