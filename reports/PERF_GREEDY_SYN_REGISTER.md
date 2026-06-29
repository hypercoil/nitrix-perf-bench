# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T21:52:10.158572+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | ok | 1.606 s / 1.660 s | 9.301 s | 467 MB (rss) | n/a (no oracle) | 0.80x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | ok | 4.205 s / 4.241 s | 6.514 s | 467 MB (rss) | n/a (no oracle) | 2.08x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 2.021 s / 2.094 s | 6.327 s | 745 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi` | ok | 1.920 s / 1.940 s | 4.371 s | 683 MB (rss) | n/a (no oracle) | 0.95x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi-autodiff` | ok | 1.504 s / 1.543 s | 4.204 s | 696 MB (rss) | n/a (no oracle) | 0.74x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax` | ok | 20.29 ms / 20.30 ms | 7.127 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi` | ok | 23.90 ms / 23.96 ms | 6.211 s | 291.85 MB (hbm) | n/a (no oracle) | 1.18x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=40 | `nitrix-jax-mi-autodiff` | ok | 31.66 ms / 31.85 ms | 6.805 s | 291.85 MB (hbm) | n/a (no oracle) | 1.56x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | ok | 1.607 s / 1.665 s | 5.786 s | 467 MB (rss) | n/a (no oracle) | 0.36x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | ok | 10.413 s / 10.457 s | 11.776 s | 467 MB (rss) | n/a (no oracle) | 2.33x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 4.478 s / 4.540 s | 14.212 s | 928 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi` | ok | 4.210 s / 4.224 s | 10.218 s | 835 MB (rss) | n/a (no oracle) | 0.94x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 3.391 s / 3.446 s | 9.626 s | 812 MB (rss) | n/a (no oracle) | 0.76x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax` | ok | 61.37 ms / 61.39 ms | 18.606 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi` | ok | 68.69 ms / 69.14 ms | 17.086 s | 291.85 MB (hbm) | n/a (no oracle) | 1.12x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=80 | `nitrix-jax-mi-autodiff` | ok | 85.36 ms / 85.76 ms | 16.735 s | 291.85 MB (hbm) | n/a (no oracle) | 1.39x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | ok | 1.602 s / 1.634 s | 5.789 s | 467 MB (rss) | n/a (no oracle) | 0.34x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | ok | 10.596 s / 10.659 s | 11.804 s | 467 MB (rss) | n/a (no oracle) | 2.27x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 4.658 s / 4.684 s | 18.326 s | 1060 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi` | ok | 4.280 s / 4.321 s | 13.514 s | 953 MB (rss) | n/a (no oracle) | 0.92x vs nitrix-jax |
| greedy_syn_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi-autodiff` | ok | 3.419 s / 3.545 s | 12.613 s | 967 MB (rss) | n/a (no oracle) | 0.73x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `ants.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `dipy.registration` | skipped | — | — | — | — | — |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax` | ok | 71.35 ms / 71.38 ms | 28.767 s | 291.85 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi` | ok | 80.37 ms / 80.62 ms | 22.884 s | 291.85 MB (hbm) | n/a (no oracle) | 1.13x vs nitrix-jax |
| greedy_syn_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=80 | `nitrix-jax-mi-autodiff` | ok | 98.13 ms / 98.75 ms | 25.835 s | 291.85 MB (hbm) | n/a (no oracle) | 1.38x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

