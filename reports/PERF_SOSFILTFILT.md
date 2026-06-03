# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: 8259a204821af34c01e8092f9cf37275f388413d | bench: ed373831de10021fb421d6197a04918a14402586
- Linux-6.1.170-213.321.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-06-03T00:26:21.874847+00:00

### Platforms

- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2
- **jax-cuda12** — NVIDIA L4 (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=2

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| sosfiltfilt | jax-cpu | channels=1024,obs=4096 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=1024,obs=4096 | `nitrix-jax` | ok | 86.53 ms / 94.15 ms | 467.84 ms | 1415 MB (rss) | ✓ 0.0022×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=1024,obs=4096 | `scipy.signal.sosfiltfilt` | ok | 60.55 ms / 63.48 ms | 82.78 ms | 1415 MB (rss) | ✓ 0.0026×tol | 0.70x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=1024,obs=4096 | `cupyx.scipy.signal.sosfiltfilt` | ok | 8.42 ms / 8.54 ms | 2.811 s | 16.78 MB (hbm) | ✓ 0.024×tol | 0.03x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=1024,obs=4096 | `nitrix-jax` | ok | 299.08 ms / 302.74 ms | 802.08 ms | 318.77 MB (hbm) | ✓ 0.0026×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=1024,obs=4096 | `scipy.signal.sosfiltfilt` | ok | 65.31 ms / 77.28 ms | 105.74 ms | 16.78 MB (hbm) | ✓ 0.0026×tol | 0.22x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=2048,obs=8192 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=2048,obs=8192 | `nitrix-jax` | ok | 381.57 ms / 437.64 ms | 782.44 ms | 1510 MB (rss) | ✓ 0.002×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=2048,obs=8192 | `scipy.signal.sosfiltfilt` | ok | 323.27 ms / 329.68 ms | 342.25 ms | 1415 MB (rss) | ✓ 0.0031×tol | 0.85x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=2048,obs=8192 | `cupyx.scipy.signal.sosfiltfilt` | ok | 26.82 ms / 27.84 ms | 2.743 s | 67.11 MB (hbm) | ✓ 0.024×tol | 0.05x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=2048,obs=8192 | `nitrix-jax` | ok | 590.00 ms / 605.65 ms | 1.070 s | 809.34 MB (hbm) | ✓ 0.0031×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=2048,obs=8192 | `scipy.signal.sosfiltfilt` | ok | 321.25 ms / 324.43 ms | 317.54 ms | 67.11 MB (hbm) | ✓ 0.0031×tol | 0.54x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=512,obs=2048 | `cupyx.scipy.signal.sosfiltfilt` | skipped | — | — | — | — | — |
| sosfiltfilt | jax-cpu | channels=512,obs=2048 | `nitrix-jax` | ok | 18.29 ms / 18.73 ms | 314.46 ms | 1415 MB (rss) | ✓ 0.0022×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cpu | channels=512,obs=2048 | `scipy.signal.sosfiltfilt` | ok | 14.94 ms / 17.63 ms | 19.28 ms | 1415 MB (rss) | ✓ 0.0025×tol | 0.82x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=512,obs=2048 | `cupyx.scipy.signal.sosfiltfilt` | ok | 5.57 ms / 5.71 ms | 26.133 s | 4.19 MB (hbm) | ✓ 0.022×tol | 0.04x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=512,obs=2048 | `nitrix-jax` | ok | 147.61 ms / 149.52 ms | 642.41 ms | 79.69 MB (hbm) | ✓ 0.0025×tol | 1.00x vs nitrix-jax |
| sosfiltfilt | jax-cuda12 | channels=512,obs=2048 | `scipy.signal.sosfiltfilt` | ok | 17.33 ms / 19.59 ms | 16.91 ms | 4.19 MB (hbm) | ✓ 0.0025×tol | 0.12x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

