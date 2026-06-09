# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 38926ea2bfa0195dd61917fdc124b0a009e99382 | bench: de1e1e3b2f4c26858e872fb2ad076dd7c2ff01d6
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-09T08:33:09.847166+00:00

### Platforms

- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail / ≈ declared-approximate: reported not gated, a fidelity/speed tradeoff that keeps its ratio). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | ok | 34.54 ms / 36.38 ms | 3.000 s | 703 MB (rss) | n/a (no oracle) | 0.08x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 436.07 ms / 446.26 ms | 14.339 s | 1003 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=1,iters=10 | `nitrix-jax` | ok | 36.78 ms / 37.58 ms | 23.600 s | 386.76 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | ok | 36.13 ms / 37.61 ms | 2.244 s | 703 MB (rss) | n/a (no oracle) | 0.04x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 984.51 ms / 994.48 ms | 65.512 s | 2038 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=2,iters=20 | `nitrix-jax` | ok | 120.20 ms / 122.78 ms | 96.166 s | 386.76 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | ok | 43.95 ms / 49.65 ms | 3.562 s | 703 MB (rss) | n/a (no oracle) | — |
| affine_register | jax-cpu | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | compile_error | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `ants.registration` | skipped | — | — | — | — | — |
| affine_register | jax-cuda12 | shape=[48, 48, 48],levels=3,iters=30 | `nitrix-jax` | ok | 243.08 ms / 243.90 ms | 211.219 s | 386.76 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

