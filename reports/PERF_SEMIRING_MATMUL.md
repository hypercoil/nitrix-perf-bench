# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- nitrix: c90f6eae1488d27e013d67260d9946fbfadcdbb3 | bench: 5bcd3f45cbe911bf9fe93bac16a9d7f7b5158cc9
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-23T02:13:48.111081+00:00

### Platforms

- **jax-cuda12** — NVIDIA A10G (gpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess
- **jax-cpu** — cpu (cpu) | jax 0.10.0 | precision highest | x64 True | isolation subprocess | sched cpu_slots=1/par=1

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`. Ratios are **within-platform** (vs that platform's reference baseline).

| case | platform | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|---|
| semiring_matmul | jax-cpu | 256x256x256 (euclidean) | `naive-dense` | ok | 25.05 ms / 26.77 ms | 77.54 ms | 288 MB (rss) | ✓ 0.00014×tol | 20.78x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (euclidean) | `nitrix-jax` | ok | 1.21 ms / 1.36 ms | 75.96 ms | 285 MB (rss) | ✓ 0.00049×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (euclidean) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 256x256x256 (euclidean) | `torch-dense` | ok | 47.06 ms / 53.43 ms | 70.18 ms | 513 MB (rss) | ✓ 0.00017×tol | 39.04x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (euclidean) | `naive-dense` | ok | 174.9 µs / 181.2 µs | 45.132 s | 68.16 MB (hbm) | ✓ 0.00014×tol | 0.11x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (euclidean) | `nitrix-jax` | ok | 1.62 ms / 1.65 ms | 77.85 ms | 3.67 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (euclidean) | `nitrix-pallas` | ok | 280.0 µs / 287.7 µs | 523.69 ms | 3.67 MB (hbm) | ✓ 0.00051×tol | 0.17x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (log) | `naive-dense` | ok | 34.37 ms / 38.14 ms | 146.60 ms | 301 MB (rss) | ✓ 8.4e-05×tol | 1.67x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (log) | `nitrix-jax` | ok | 20.63 ms / 24.27 ms | 116.30 ms | 285 MB (rss) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (log) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 256x256x256 (log) | `torch-dense` | ok | 61.75 ms / 65.86 ms | 116.66 ms | 519 MB (rss) | ✓ 9.4e-05×tol | 2.99x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (log) | `naive-dense` | ok | 203.4 µs / 208.6 µs | 310.033 s | 85.20 MB (hbm) | ✓ 7.4e-05×tol | 0.14x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (log) | `nitrix-jax` | ok | 1.45 ms / 1.47 ms | 103.21 ms | 5.83 MB (hbm) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (log) | `nitrix-pallas` | ok | 1.76 ms / 1.77 ms | 2.053 s | 5.83 MB (hbm) | ✓ 0.00017×tol | 1.21x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (real) | `jnp-matmul` | ok | 259.2 µs / 302.7 µs | 12.57 ms | 285 MB (rss) | ✓ 0.12×tol | 0.21x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (real) | `naive-dense` | ok | 25.25 ms / 29.64 ms | 59.85 ms | 285 MB (rss) | ✓ 0.051×tol | 20.85x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (real) | `nitrix-jax` | ok | 1.21 ms / 1.28 ms | 65.76 ms | 285 MB (rss) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (real) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 256x256x256 (real) | `torch-dense` | ok | 24.00 ms / 25.22 ms | 24.66 ms | 444 MB (rss) | ✓ 0.037×tol | 19.81x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (real) | `jnp-matmul` | ok | 137.8 µs / 140.2 µs | 149.69 ms | 72.35 MB (hbm) | ✓ 0.069×tol | 0.09x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (real) | `naive-dense` | ok | 161.9 µs / 166.7 µs | 308.216 s | 68.16 MB (hbm) | ✓ 0.027×tol | 0.11x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (real) | `nitrix-jax` | ok | 1.48 ms / 1.68 ms | 71.79 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (real) | `nitrix-pallas` | ok | 269.3 µs / 270.8 µs | 476.99 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 0.18x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (tropical_max_plus) | `naive-dense` | ok | 24.35 ms / 25.86 ms | 68.75 ms | 285 MB (rss) | ✓ 5.8e-05×tol | 5.17x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (tropical_max_plus) | `nitrix-jax` | ok | 4.71 ms / 5.50 ms | 72.97 ms | 285 MB (rss) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 256x256x256 (tropical_max_plus) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 256x256x256 (tropical_max_plus) | `torch-dense` | ok | 25.21 ms / 25.69 ms | 31.43 ms | 445 MB (rss) | ✓ 5.8e-05×tol | 5.36x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (tropical_max_plus) | `naive-dense` | ok | 162.5 µs / 167.2 µs | 301.032 s | 68.16 MB (hbm) | ✓ 5.8e-05×tol | 0.10x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (tropical_max_plus) | `nitrix-jax` | ok | 1.60 ms / 1.62 ms | 68.54 ms | 2.62 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 256x256x256 (tropical_max_plus) | `nitrix-pallas` | ok | 267.2 µs / 282.6 µs | 444.38 ms | 2.62 MB (hbm) | ✓ 5.8e-05×tol | 0.17x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (euclidean) | `naive-dense` | ok | 191.37 ms / 192.34 ms | 218.20 ms | 764 MB (rss) | ✓ 0.00018×tol | 29.49x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (euclidean) | `nitrix-jax` | ok | 6.49 ms / 7.51 ms | 69.12 ms | 285 MB (rss) | ✓ 0.0008×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (euclidean) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 512x512x512 (euclidean) | `torch-dense` | ok | 369.50 ms / 374.11 ms | 373.54 ms | 1427 MB (rss) | ✓ 0.00018×tol | 56.93x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (euclidean) | `naive-dense` | ok | 401.6 µs / 411.6 µs | 263.698 s | 71.30 MB (hbm) | ✓ 0.00017×tol | 0.07x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (euclidean) | `nitrix-jax` | ok | 5.73 ms / 6.29 ms | 94.97 ms | 14.68 MB (hbm) | ✓ 0.0008×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (euclidean) | `nitrix-pallas` | ok | 422.7 µs / 433.0 µs | 522.65 ms | 14.68 MB (hbm) | ✓ 0.0008×tol | 0.07x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (log) | `naive-dense` | ok | 251.46 ms / 259.31 ms | 324.34 ms | 775 MB (rss) | ✓ 8.2e-05×tol | 1.76x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (log) | `nitrix-jax` | ok | 143.05 ms / 146.25 ms | 249.16 ms | 285 MB (rss) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (log) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 512x512x512 (log) | `torch-dense` | ok | 479.26 ms / 483.89 ms | 501.45 ms | 1430 MB (rss) | ✓ 7.4e-05×tol | 3.35x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (log) | `naive-dense` | ok | 672.2 µs / 679.3 µs | 583.626 s | 73.40 MB (hbm) | ✓ 0.0001×tol | 0.12x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (log) | `nitrix-jax` | ok | 5.81 ms / 6.27 ms | 131.26 ms | 23.33 MB (hbm) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (log) | `nitrix-pallas` | ok | 3.42 ms / 3.43 ms | 2.046 s | 23.33 MB (hbm) | ✓ 0.00025×tol | 0.59x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (real) | `jnp-matmul` | ok | 1.48 ms / 1.59 ms | 16.26 ms | 285 MB (rss) | ✓ 0.24×tol | 0.20x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (real) | `naive-dense` | ok | 194.54 ms / 204.08 ms | 200.87 ms | 755 MB (rss) | ✓ 0.06×tol | 26.79x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (real) | `nitrix-jax` | ok | 7.26 ms / 8.05 ms | 79.36 ms | 285 MB (rss) | ✓ 0.24×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (real) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 512x512x512 (real) | `torch-dense` | ok | 192.48 ms / 196.32 ms | 210.76 ms | 901 MB (rss) | ✓ 0.082×tol | 26.51x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (real) | `jnp-matmul` | ok | 121.9 µs / 146.7 µs | 101.19 ms | 79.69 MB (hbm) | ✓ 0.13×tol | 0.02x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (real) | `naive-dense` | ok | 445.9 µs / 454.5 µs | 285.612 s | 71.30 MB (hbm) | ✓ 0.066×tol | 0.08x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (real) | `nitrix-jax` | ok | 5.77 ms / 6.46 ms | 81.73 ms | 10.49 MB (hbm) | ✓ 0.17×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (real) | `nitrix-pallas` | ok | 403.2 µs / 407.5 µs | 442.54 ms | 10.49 MB (hbm) | ✓ 0.17×tol | 0.07x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (tropical_max_plus) | `naive-dense` | ok | 185.61 ms / 188.00 ms | 175.59 ms | 757 MB (rss) | ✓ 5.8e-05×tol | 4.19x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (tropical_max_plus) | `nitrix-jax` | ok | 44.34 ms / 69.56 ms | 98.58 ms | 285 MB (rss) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cpu | 512x512x512 (tropical_max_plus) | `nitrix-pallas` | skipped | — | — | — | — | — |
| semiring_matmul | jax-cpu | 512x512x512 (tropical_max_plus) | `torch-dense` | ok | 200.90 ms / 201.74 ms | 202.70 ms | 903 MB (rss) | ✓ 5.8e-05×tol | 4.53x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (tropical_max_plus) | `naive-dense` | ok | 400.1 µs / 406.0 µs | 292.271 s | 71.30 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (tropical_max_plus) | `nitrix-jax` | ok | 5.78 ms / 6.27 ms | 81.32 ms | 10.49 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | jax-cuda12 | 512x512x512 (tropical_max_plus) | `nitrix-pallas` | ok | 395.2 µs / 400.1 µs | 447.38 ms | 10.49 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

