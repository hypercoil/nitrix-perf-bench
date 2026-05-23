# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- device: NVIDIA A10G (gpu)
- jax: 0.10.0 | backend: gpu
- precision: highest | x64: True | preallocate: false | compile_cache: disabled | isolation: subprocess
- nitrix: c90f6eae1488d27e013d67260d9946fbfadcdbb3 | bench: 5bcd3f45cbe911bf9fe93bac16a9d7f7b5158cc9
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-23T02:13:48.111081+00:00

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`.

| case | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|
| semiring_matmul | 256x256x256 (real) | `nitrix-jax` | ok | 1.48 ms / 1.68 ms | 71.79 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `nitrix-pallas` | ok | 269.3 µs / 270.8 µs | 476.99 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 0.18x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `naive-dense` | ok | 161.9 µs / 166.7 µs | 308.216 s | 68.16 MB (hbm) | ✓ 0.027×tol | 0.11x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `jnp-matmul` | ok | 137.8 µs / 140.2 µs | 149.69 ms | 72.35 MB (hbm) | ✓ 0.069×tol | 0.09x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `nitrix-jax` | ok | 1.45 ms / 1.47 ms | 103.21 ms | 5.83 MB (hbm) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `nitrix-pallas` | ok | 1.76 ms / 1.77 ms | 2.053 s | 5.83 MB (hbm) | ✓ 0.00017×tol | 1.21x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `naive-dense` | ok | 203.4 µs / 208.6 µs | 310.033 s | 85.20 MB (hbm) | ✓ 7.4e-05×tol | 0.14x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `nitrix-jax` | ok | 1.60 ms / 1.62 ms | 68.54 ms | 2.62 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `nitrix-pallas` | ok | 267.2 µs / 282.6 µs | 444.38 ms | 2.62 MB (hbm) | ✓ 5.8e-05×tol | 0.17x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `naive-dense` | ok | 162.5 µs / 167.2 µs | 301.032 s | 68.16 MB (hbm) | ✓ 5.8e-05×tol | 0.10x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `nitrix-jax` | ok | 1.62 ms / 1.65 ms | 77.85 ms | 3.67 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `nitrix-pallas` | ok | 280.0 µs / 287.7 µs | 523.69 ms | 3.67 MB (hbm) | ✓ 0.00051×tol | 0.17x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `naive-dense` | ok | 174.9 µs / 181.2 µs | 45.132 s | 68.16 MB (hbm) | ✓ 0.00014×tol | 0.11x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `nitrix-jax` | ok | 5.77 ms / 6.46 ms | 81.73 ms | 10.49 MB (hbm) | ✓ 0.17×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `nitrix-pallas` | ok | 403.2 µs / 407.5 µs | 442.54 ms | 10.49 MB (hbm) | ✓ 0.17×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `naive-dense` | ok | 445.9 µs / 454.5 µs | 285.612 s | 71.30 MB (hbm) | ✓ 0.066×tol | 0.08x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `jnp-matmul` | ok | 121.9 µs / 146.7 µs | 101.19 ms | 79.69 MB (hbm) | ✓ 0.13×tol | 0.02x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `nitrix-jax` | ok | 5.81 ms / 6.27 ms | 131.26 ms | 23.33 MB (hbm) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `nitrix-pallas` | ok | 3.42 ms / 3.43 ms | 2.046 s | 23.33 MB (hbm) | ✓ 0.00025×tol | 0.59x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `naive-dense` | ok | 672.2 µs / 679.3 µs | 583.626 s | 73.40 MB (hbm) | ✓ 0.0001×tol | 0.12x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `nitrix-jax` | ok | 5.78 ms / 6.27 ms | 81.32 ms | 10.49 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `nitrix-pallas` | ok | 395.2 µs / 400.1 µs | 447.38 ms | 10.49 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `naive-dense` | ok | 400.1 µs / 406.0 µs | 292.271 s | 71.30 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `nitrix-jax` | ok | 5.73 ms / 6.29 ms | 94.97 ms | 14.68 MB (hbm) | ✓ 0.0008×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `nitrix-pallas` | ok | 422.7 µs / 433.0 µs | 522.65 ms | 14.68 MB (hbm) | ✓ 0.0008×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `naive-dense` | ok | 401.6 µs / 411.6 µs | 263.698 s | 71.30 MB (hbm) | ✓ 0.00017×tol | 0.07x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt ran in its own process, so the high-water mark *is* this attempt's peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE §B).

