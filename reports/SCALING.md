# Scaling / crossover report

Scale-gaming defence: the scaling curve + the stated cost law, so a small-size win cannot hide a large-size / batched loss or OOM. Platform: `jax-cuda12`.

## pca_fit  (nitrix.stats.pca_fit)  [jax-cuda12]

**Cost law.** cov is O(n * d^2) (one BLAS matmul); the eigh of the (d, d) cov is O(d^3) and dominates at brain-feature d. HBM ~ d^2 (the cov). MEASURED (L4): the cuSOLVER eigh stayed GPU-native through d=2048 in fresh workers (NO CPU fallback fired; the older d>=256 routing did not reproduce), so nitrix is at PARITY with the cupy device-eigh on GPU (~0.93-0.96x; cupy marginally faster, 0.63x at tiny d=128 where nitrix fixed overhead dominates). The WIN is on CPU vs sklearn (6-12x): nitrix eigh-decomposes the (d, d) cov where sklearn SVDs the (n, d) data -- structurally cheaper when n>d. GPU-vs-CPU for nitrix is ~30x at d=2048. The size tier varies d to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 2048x128 k16 | 1.57ms | 1.00ms (eigh_cov) | 1.58x | 74.4MB | 1.0MB | 71x |
| 2048x256 k16 | 2.05ms | 1.97ms (eigh_cov) | 1.04x | 77.6MB | 2.1MB | 37x |
| 1024x512 k16 | 4.51ms | 4.24ms (eigh_cov) | 1.06x | 77.6MB | 2.1MB | 37x |
| 4096x1024 k32 | 12.11ms | 11.43ms (eigh_cov) | 1.06x | 96.5MB | 16.8MB | 6x |
| 8192x2048 k32 | 44.92ms | 41.59ms (eigh_cov) | 1.08x | 373.3MB | 67.1MB | 6x |

- **Speed:** nitrix wins 0/5 sizes; baseline ahead at `2048x128 k16` 1.58x, `8192x2048 k32` 1.08x, `1024x512 k16` 1.06x, `4096x1024 k32` 1.06x (+1 more); at the largest `8192x2048 k32`, baseline 1.08x ahead.
- **Projected OOM (≈24GB):** nitrix ~34516.8 Melem vs best baseline ~192000 Melem (~6x more headroom).

## pca_inverse_transform  (nitrix.stats.pca_inverse_transform)  [jax-cuda12]

**Cost law.** Z @ components + mean: O(n * d * k) -- one BLAS-class matmul, the GPU-friendly regime (no eigh, so no cuSOLVER fallback; contrast pca_fit). HBM ~ n*d (the output dominates) + n*k (input). The size tier varies n to whole-brain voxel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 8192x256 k32 | 0.12ms | 0.05ms (matmul) | 2.17x | 105.9MB | 1.1MB | 98x |
| 16384x512 k64 | 0.24ms | 0.46ms (matmul) | 0.52x | 176.3MB | 4.3MB | 41x |
| 8192x1024 k64 | 0.25ms | 0.47ms (matmul) | 0.54x | 174.3MB | 2.4MB | 74x |
| 65536x1024 k64 | 1.30ms | 3.56ms (matmul) | 0.37x | 625.5MB | 17.0MB | 37x |
| 131072x512 k64 | 1.41ms | 3.59ms (matmul) | 0.39x | 692.1MB | 33.7MB | 21x |

- **Speed:** nitrix wins 4/5 sizes; baseline ahead at `8192x256 k32` 2.17x; at the largest `131072x512 k64`, nitrix 2.55x ahead.
- **Projected OOM (≈24GB):** nitrix ~148945.0 Melem vs best baseline ~3059861 Melem (~21x more headroom).

## pca_transform  (nitrix.stats.pca_transform)  [jax-cuda12]

**Cost law.** (X - mean) @ components^T: O(n * d * k) -- one BLAS-class matmul, the GPU-friendly regime (no eigh, so no cuSOLVER fallback; contrast pca_fit). HBM ~ n*d (the input dominates) + n*k (output). The size tier varies n to whole-brain voxel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 8192x256 k32 | 0.13ms | 0.06ms (matmul) | 2.16x | 83.9MB | 8.4MB | 10x |
| 16384x512 k64 | 0.46ms | 0.36ms (matmul) | 1.29x | 146.8MB | 33.7MB | 4x |
| 8192x1024 k64 | 0.42ms | 0.34ms (matmul) | 1.22x | 142.6MB | 33.8MB | 4x |
| 65536x1024 k64 | 3.70ms | 3.58ms (matmul) | 1.04x | 608.7MB | 268.7MB | 2x |
| 131072x512 k64 | 3.73ms | 3.62ms (matmul) | 1.03x | 642.0MB | 268.6MB | 2x |

- **Speed:** nitrix wins 0/5 sizes; baseline ahead at `8192x256 k32` 2.16x, `16384x512 k64` 1.29x, `8192x1024 k64` 1.22x, `65536x1024 k64` 1.04x (+1 more); at the largest `65536x1024 k64`, baseline 1.04x ahead.
- **Projected OOM (≈24GB):** nitrix ~160561.3 Melem vs best baseline ~383620 Melem (~2x more headroom).

