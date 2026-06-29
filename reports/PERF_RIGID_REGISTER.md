# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T21:20:20.563563+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 1.248 s / 1.281 s | 12.163 s | 469 MB (rss) | n/a (no oracle) | 20.23x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | ok | 524.70 ms / 531.74 ms | 2.624 s | 469 MB (rss) | n/a (no oracle) | 8.50x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 61.70 ms / 63.76 ms | 1.121 s | 629 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 1.88 ms / 1.90 ms | 4.596 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 1.236 s / 1.341 s | 8.469 s | 469 MB (rss) | n/a (no oracle) | 13.57x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | ok | 1.379 s / 1.401 s | 3.317 s | 469 MB (rss) | n/a (no oracle) | 15.14x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 91.06 ms / 92.84 ms | 3.101 s | 721 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 3.77 ms / 3.79 ms | 11.791 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 1.268 s / 1.376 s | 9.864 s | 469 MB (rss) | n/a (no oracle) | 13.58x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | ok | 1.187 s / 1.196 s | 2.889 s | 469 MB (rss) | n/a (no oracle) | 12.72x vs nitrix-jax |
| rigid_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 93.37 ms / 95.77 ms | 4.908 s | 790 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `dipy.registration` | skipped | — | — | — | — | — |
| rigid_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 4.86 ms / 4.90 ms | 17.034 s | 203.42 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

