# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: ff2bf24d06cb02088c5cb43ad62795a3705b0a56 | bench: c8c5b5033f3a8aa8dde9a2ea33e5917ad22d3292
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-02T04:01:38.958585+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| precision | jax-cpu | c=128,obs=1024 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=128,obs=1024 | `nitrix-jax` | ok | 718.5 µs / 788.5 µs | 264.43 ms | 578 MB (rss) | ✓ 0.0014×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=128,obs=1024 | `numpy.inv_cov` | ok | 739.0 µs / 752.0 µs | 1.21 ms | 566 MB (rss) | ✓ 9.9e-05×tol | 1.03x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `cupy.inv_cov` | ok | 715.8 µs / 724.3 µs | 857.33 ms | 0.52 MB (hbm) | ✓ 0.00072×tol | 2.15x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `nitrix-jax` | ok | 332.5 µs / 338.5 µs | 594.92 ms | 72.88 MB (hbm) | ✓ 0.00075×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=128,obs=1024 | `numpy.inv_cov` | ok | 757.5 µs / 8.85 ms | 911.8 µs | 0.52 MB (hbm) | ✓ 9.9e-05×tol | 2.28x vs nitrix-jax |
| precision | jax-cpu | c=256,obs=2048 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=256,obs=2048 | `nitrix-jax` | ok | 4.73 ms / 8.11 ms | 297.55 ms | 596 MB (rss) | ✓ 0.0017×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=256,obs=2048 | `numpy.inv_cov` | ok | 4.61 ms / 6.64 ms | 4.89 ms | 566 MB (rss) | ✓ 0.0001×tol | 0.97x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `cupy.inv_cov` | ok | 1.36 ms / 1.37 ms | 228.30 ms | 2.10 MB (hbm) | ✓ 0.00077×tol | 2.24x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `nitrix-jax` | ok | 605.3 µs / 607.2 µs | 616.81 ms | 77.59 MB (hbm) | ✓ 0.00081×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=256,obs=2048 | `numpy.inv_cov` | ok | 3.90 ms / 54.91 ms | 1.105 s | 2.10 MB (hbm) | ✓ 0.0001×tol | 6.45x vs nitrix-jax |
| precision | jax-cpu | c=512,obs=4096 | `cupy.inv_cov` | skipped | — | — | — | — | — |
| precision | jax-cpu | c=512,obs=4096 | `nitrix-jax` | ok | 23.58 ms / 31.99 ms | 2.222 s | 667 MB (rss) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| precision | jax-cpu | c=512,obs=4096 | `numpy.inv_cov` | ok | 35.46 ms / 59.03 ms | 32.58 ms | 566 MB (rss) | ✓ 0.00011×tol | 1.50x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `cupy.inv_cov` | ok | 6.80 ms / 6.83 ms | 260.48 ms | 8.39 MB (hbm) | ✓ 0.0011×tol | 4.27x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `nitrix-jax` | ok | 1.59 ms / 1.60 ms | 880.26 ms | 88.08 MB (hbm) | ✓ 0.0012×tol | 1.00x vs nitrix-jax |
| precision | jax-cuda12 | c=512,obs=4096 | `numpy.inv_cov` | ok | 23.77 ms / 27.38 ms | 24.24 ms | 8.39 MB (hbm) | ✓ 0.00011×tol | 14.94x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

