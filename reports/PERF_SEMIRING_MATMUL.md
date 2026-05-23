# nitrix-perf-bench results

> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); schema_version 1 (frozen). No values are hand-edited.

## Host

- device: NVIDIA A10G (gpu)
- jax: 0.10.0 | backend: gpu
- precision: highest | x64: True | preallocate: false | compile_cache: disabled
- nitrix: c90f6eae1488d27e013d67260d9946fbfadcdbb3 | bench: b9ba402b301276f1a6c2c1614faf6db1169288b7
- Linux-6.1.161-183.298.amzn2023.x86_64-x86_64-with-glibc2.39 | python 3.13.13 | 2026-05-23T00:05:42.342470+00:00

## Measurements

`steady` = post-warm-up min / median; `compile` = cold first call; `fidelity` = worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row keeps its measurements but its ratio is `refused`.

| case | param | baseline | status | steady (min/med) | compile | mem | fidelity | ratio |
|---|---|---|---|---|---|---|---|---|
| semiring_matmul | 256x256x256 (real) | `nitrix-jax` | ok | 1.48 ms / 1.61 ms | 69.60 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `nitrix-pallas` | ok | 269.6 µs / 274.9 µs | 466.95 ms | 2.62 MB (hbm) | ✓ 0.12×tol | 0.18x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `naive-dense` | ok | 139.9 µs / 145.6 µs | 308.602 s | 68.16 MB (hbm) | ✓ 0.027×tol | 0.09x vs nitrix-jax |
| semiring_matmul | 256x256x256 (real) | `jnp-matmul` | ok | 153.4 µs / 154.6 µs | 184.33 ms | 72.35 MB (hbm) | ✓ 0.069×tol | 0.10x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `nitrix-jax` | ok | 1.45 ms / 1.48 ms | 103.29 ms | 72.35 MB (hbm) | ✓ 0.00017×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `nitrix-pallas` | ok | 1.77 ms / 1.78 ms | 1.855 s | 72.35 MB (hbm) | ✓ 0.00017×tol | 1.22x vs nitrix-jax |
| semiring_matmul | 256x256x256 (log) | `naive-dense` | ok | 202.8 µs / 206.4 µs | 317.007 s | 85.20 MB (hbm) | ✓ 7.4e-05×tol | 0.14x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `nitrix-jax` | ok | 1.45 ms / 1.47 ms | 69.44 ms | 85.20 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `nitrix-pallas` | ok | 276.3 µs / 295.7 µs | 261.48 ms | 85.20 MB (hbm) | ✓ 5.8e-05×tol | 0.19x vs nitrix-jax |
| semiring_matmul | 256x256x256 (tropical_max_plus) | `naive-dense` | ok | 119.0 µs / 119.7 µs | 69.20 ms | 85.20 MB (hbm) | ✓ 5.8e-05×tol | 0.08x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `nitrix-jax` | ok | 1.46 ms / 1.50 ms | 77.38 ms | 85.20 MB (hbm) | ✓ 0.00051×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `nitrix-pallas` | ok | 289.9 µs / 292.8 µs | 340.31 ms | 85.20 MB (hbm) | ✓ 0.00051×tol | 0.20x vs nitrix-jax |
| semiring_matmul | 256x256x256 (euclidean) | `naive-dense` | ok | 122.3 µs / 123.7 µs | 45.764 s | 85.20 MB (hbm) | ✓ 0.00014×tol | 0.08x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `nitrix-jax` | ok | 5.71 ms / 5.85 ms | 83.14 ms | 85.20 MB (hbm) | ✓ 0.17×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `nitrix-pallas` | ok | 398.3 µs / 410.2 µs | 259.84 ms | 85.20 MB (hbm) | ✓ 0.17×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `naive-dense` | ok | 386.5 µs / 392.4 µs | 294.208 s | 85.20 MB (hbm) | ✓ 0.066×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (real) | `jnp-matmul` | ok | 166.2 µs / 170.5 µs | 18.49 ms | 85.20 MB (hbm) | ✓ 0.13×tol | 0.03x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `nitrix-jax` | ok | 5.78 ms / 6.01 ms | 183.27 ms | 85.20 MB (hbm) | ✓ 0.00025×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `nitrix-pallas` | ok | 3.40 ms / 3.42 ms | 1.867 s | 85.20 MB (hbm) | ✓ 0.00025×tol | 0.59x vs nitrix-jax |
| semiring_matmul | 512x512x512 (log) | `naive-dense` | ok | 646.6 µs / 680.8 µs | 586.821 s | 89.13 MB (hbm) | ✓ 0.0001×tol | 0.11x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `nitrix-jax` | ok | 5.74 ms / 6.00 ms | 83.15 ms | 89.13 MB (hbm) | ✓ 5.8e-05×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `nitrix-pallas` | ok | 395.3 µs / 403.4 µs | 262.55 ms | 89.13 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (tropical_max_plus) | `naive-dense` | ok | 395.6 µs / 399.2 µs | 74.31 ms | 89.13 MB (hbm) | ✓ 5.8e-05×tol | 0.07x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `nitrix-jax` | ok | 5.71 ms / 6.19 ms | 97.68 ms | 89.13 MB (hbm) | ✓ 0.0008×tol | 1.00x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `nitrix-pallas` | ok | 436.9 µs / 444.9 µs | 341.82 ms | 89.13 MB (hbm) | ✓ 0.0008×tol | 0.08x vs nitrix-jax |
| semiring_matmul | 512x512x512 (euclidean) | `naive-dense` | ok | 398.9 µs / 405.0 µs | 263.635 s | 89.13 MB (hbm) | ✓ 0.00017×tol | 0.07x vs nitrix-jax |

## Notes

- `steady` is the post-warm-up min / median of the timed loop; `ratio` is on `min` vs the reference baseline, computed and stored in L1 (this renderer does no metric arithmetic).
- `compile` is the **cold** first-call cost (`jax.clear_caches()` per attempt; persistent cache disabled) — what a user pays once, not a steady-state number.
- `fidelity` is `rel_to_tol`: the worst error as a multiple of the allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is tolerance-relative on purpose — a bare relative error is meaningless for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).
- **`mem` (peak_hbm) is a process-wide high-water mark.** XLA's `peak_bytes_in_use` does not reset between attempts in this in-process driver, so it only ever rises: once one attempt allocates a large buffer, later rows inherit that floor. Read the *jumps* (they attribute to the attempt that caused them), not the absolute per-row value. True per-attempt isolation arrives with the P1 subprocess workers (annex §B).

