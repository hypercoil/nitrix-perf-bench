# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T21:24:14.384052+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 1.270 s / 1.364 s | 6.474 s | 469 MB (rss) | n/a (no oracle) | 21.74x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | ok | 546.63 ms / 576.01 ms | 1.708 s | 469 MB (rss) | n/a (no oracle) | 9.36x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 58.40 ms / 59.97 ms | 3.127 s | 692 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 5.96 ms / 5.99 ms | 6.802 s | 470.65 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 1.213 s / 1.282 s | 5.718 s | 469 MB (rss) | n/a (no oracle) | 10.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 1.235 s / 1.267 s | 2.240 s | 469 MB (rss) | n/a (no oracle) | 10.19x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 121.26 ms / 123.72 ms | 5.506 s | 789 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 10.56 ms / 10.58 ms | 12.577 s | 470.65 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 1.233 s / 1.405 s | 5.989 s | 469 MB (rss) | n/a (no oracle) | 9.97x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | ok | 1.413 s / 1.435 s | 2.409 s | 469 MB (rss) | n/a (no oracle) | 11.43x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 123.64 ms / 127.75 ms | 5.476 s | 805 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 11.68 ms / 11.88 ms | 12.114 s | 470.65 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

