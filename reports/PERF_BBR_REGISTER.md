# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 92ec5fca2b5689f7f3adc05934c5c897c4110bb0 | bench: dd6ba100ec082f0431d522cb535822ba0f252755
- Linux-6.1.172-216.329.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-29T21:51:37.944181+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 139.40 ms / 154.04 ms | 2.382 s | 797 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=100 | `nitrix-jax` | ok | 16.37 ms / 16.53 ms | 5.799 s | 185.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cpu | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 106.10 ms / 114.44 ms | 2.235 s | 808 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| bbr_register | jax-cuda12 | shape=[48, 48, 48],N=2000,iters=50 | `nitrix-jax` | ok | 9.18 ms / 9.24 ms | 5.654 s | 185.04 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

