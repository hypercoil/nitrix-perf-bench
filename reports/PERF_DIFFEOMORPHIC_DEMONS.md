# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T22:11:35.519811+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | ok | 1.713 s / 1.727 s | 6.491 s | 469 MB (rss) | n/a (no oracle) | 4.87x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | ok | 2.601 s / 2.706 s | 3.861 s | 469 MB (rss) | n/a (no oracle) | 7.40x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 351.72 ms / 371.89 ms | 1.636 s | 646 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax-algebra` | ok | 391.93 ms / 398.49 ms | 1.436 s | 652 MB (rss) | n/a (no oracle) | 1.11x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 284.95 ms / 302.33 ms | 359.67 ms | 585 MB (rss) | n/a (no oracle) | 0.81x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax` | ok | 3.41 ms / 3.42 ms | 4.510 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `nitrix-jax-algebra` | ok | 4.26 ms / 4.30 ms | 3.648 s | 203.42 MB (hbm) | n/a (no oracle) | 1.25x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=20 | `simpleitk.demons` | ok | 300.87 ms / 325.55 ms | 2.870 s | 0.88 MB (hbm) | n/a (no oracle) | 88.20x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 1.718 s / 1.776 s | 5.842 s | 469 MB (rss) | n/a (no oracle) | 4.15x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 3.754 s / 3.793 s | 4.700 s | 469 MB (rss) | n/a (no oracle) | 9.06x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 414.24 ms / 429.11 ms | 4.245 s | 753 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 422.75 ms / 432.24 ms | 3.932 s | 738 MB (rss) | n/a (no oracle) | 1.02x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 282.89 ms / 307.11 ms | 350.65 ms | 583 MB (rss) | n/a (no oracle) | 0.68x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 4.61 ms / 4.62 ms | 12.636 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax-algebra` | ok | 6.62 ms / 6.74 ms | 9.784 s | 203.42 MB (hbm) | n/a (no oracle) | 1.44x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `simpleitk.demons` | ok | 303.16 ms / 329.24 ms | 361.15 ms | 0.88 MB (hbm) | n/a (no oracle) | 65.82x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | ok | 1.700 s / 1.743 s | 5.729 s | 469 MB (rss) | n/a (no oracle) | 2.05x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | ok | 8.144 s / 8.186 s | 8.964 s | 469 MB (rss) | n/a (no oracle) | 9.84x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 827.81 ms / 838.95 ms | 4.600 s | 765 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax-algebra` | ok | 812.46 ms / 819.53 ms | 3.982 s | 743 MB (rss) | n/a (no oracle) | 0.98x vs nitrix-jax |
| diffeomorphic_demons | jax-cpu | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 555.36 ms / 570.40 ms | 618.13 ms | 584 MB (rss) | n/a (no oracle) | 0.67x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax` | ok | 9.07 ms / 9.22 ms | 8.964 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `nitrix-jax-algebra` | ok | 12.12 ms / 12.28 ms | 9.573 s | 203.42 MB (hbm) | n/a (no oracle) | 1.34x vs nitrix-jax |
| diffeomorphic_demons | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=40 | `simpleitk.demons` | ok | 551.89 ms / 597.48 ms | 624.41 ms | 0.88 MB (hbm) | n/a (no oracle) | 60.87x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

