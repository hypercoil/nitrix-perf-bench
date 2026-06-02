# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: be1403238ab822dfd4d8ce256fe79dbb69bacbbf | bench: 8d7c2e44607e6159bfebb98fb9b3dfedaac147ed
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T21:39:52.519203+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| n4_bias_field_correction | jax-cpu | s=32 | `nitrix-jax` | ok | 50.64 ms / 52.89 ms | 1.175 s | 715 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cpu | s=32 | `simpleitk.N4` | ok | 1.677 s / 2.054 s | 1.999 s | 715 MB (rss) | n/a (no oracle) | 33.11x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=32 | `nitrix-jax` | ok | 16.06 ms / 16.15 ms | 3.619 s | 168.30 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=32 | `simpleitk.N4` | ok | 1.589 s / 2.073 s | 2.114 s | 0.26 MB (hbm) | n/a (no oracle) | 98.95x vs nitrix-jax |
| n4_bias_field_correction | jax-cpu | s=48 | `nitrix-jax` | ok | 161.33 ms / 169.47 ms | 1.594 s | 715 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cpu | s=48 | `simpleitk.N4` | ok | 6.276 s / 6.353 s | 5.173 s | 715 MB (rss) | n/a (no oracle) | 38.90x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=48 | `nitrix-jax` | ok | 33.01 ms / 33.09 ms | 4.140 s | 73.40 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=48 | `simpleitk.N4` | ok | 4.986 s / 6.339 s | 6.453 s | 0.88 MB (hbm) | n/a (no oracle) | 151.05x vs nitrix-jax |
| n4_bias_field_correction | jax-cpu | s=64 | `nitrix-jax` | ok | 369.19 ms / 392.46 ms | 2.059 s | 715 MB (rss) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cpu | s=64 | `simpleitk.N4` | ok | 14.597 s / 14.713 s | 12.085 s | 715 MB (rss) | n/a (no oracle) | 39.54x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=64 | `nitrix-jax` | ok | 65.08 ms / 65.23 ms | 4.326 s | 75.50 MB (hbm) | n/a (no oracle) | 1.00x vs nitrix-jax |
| n4_bias_field_correction | jax-cuda12 | s=64 | `simpleitk.N4` | ok | 14.594 s / 14.744 s | 14.959 s | 2.10 MB (hbm) | n/a (no oracle) | 224.24x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

