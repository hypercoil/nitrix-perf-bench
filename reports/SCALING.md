# Scaling / crossover report

Scale-gaming defence: the scaling curve + the stated cost law, so a small-size win cannot hide a large-size / batched loss or OOM. Platform: `jax-cuda12`.

## distance_transform  (nitrix.morphology.distance_transform)  [jax-cuda12]

**Cost law.** time nitrix O(n^(d+1))/axis (one shallow min-plus matmul) vs F-H O(n^d) (deeper sequential scan); HBM nitrix ~5-1000x the in-place F-H refs (L4). Hypothesis: GPU wall-clock depth-bound at small scale (low-depth brute force wins despite more FLOPs), flop/HBM-bound at large/batched scale (F-H wins, nitrix OOMs first). Differentiability is a bonus of the substrate, not the reason it was chosen

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 64x64 | 0.15ms | 0.20ms (distance_transform_edt) | 0.75x | 33.6MB | 0.0MB | 2051x |
| 128x128 | 0.22ms | 0.20ms (distance_transform_edt) | 1.09x | 33.8MB | 0.1MB | 515x |
| 256x256 | 0.34ms | 0.24ms (distance_transform_edt) | 1.40x | 34.3MB | 0.3MB | 131x |
| 64x64x64 | 0.25ms | 0.24ms (distance_transform_edt) | 1.04x | 36.7MB | 1.0MB | 35x |
| 512x512 | 0.56ms | 0.33ms (distance_transform_edt) | 1.70x | 36.7MB | 1.0MB | 35x |
| 128x128x128 | 0.47ms | 0.54ms (distance_transform_edt) | 0.87x | 58.7MB | 8.4MB | 7x |
| 4*128x128x128 | 2.24ms | 2.68ms (distance_transform_edt) | 0.84x | 167.8MB | 33.6MB | 5x |
| 256x256x256 | 7.31ms | 6.05ms (distance_transform_edt) | 1.21x | 335.5MB | 67.1MB | 5x |
| 8*128x128x128 | 4.96ms | 5.39ms (distance_transform_edt) | 0.92x | 335.5MB | 67.1MB | 5x |
| 16*128x128x128 | 9.76ms | 10.78ms (distance_transform_edt) | 0.91x | 671.1MB | 134.2MB | 5x |

- **Speed:** nitrix wins 5/10 sizes; baseline ahead at `512x512` 1.70x, `256x256` 1.40x, `256x256x256` 1.21x, `128x128` 1.09x (+1 more); at the largest `16*128x128x128`, nitrix 1.10x ahead.
- **Projected OOM (≈24GB):** nitrix ~1200.0 Melem vs best baseline ~6000 Melem (~5x more headroom).

