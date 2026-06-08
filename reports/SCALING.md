# Scaling / crossover report

Scale-gaming defence: the scaling curve + the stated cost law, so a small-size win cannot hide a large-size / batched loss or OOM. Platform: `jax-cuda12`.

## close  (nitrix.morphology.close)  [jax-cuda12]

**Cost law.** time: flat box O(N) (two fused reduce_windows) vs explicit SE O(N*k^d) (two im2col passes); HBM: box O(N), explicit-SE O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy (O(N*k), in-place) holds. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.09ms | 0.24ms (grey_closing) | 0.40x | 1.0MB | 0.3MB | 4x |
| 256x256 disk3 | 0.64ms | 0.53ms (grey_closing) | 1.21x | 93.1MB | 0.3MB | 355x |
| 64x64x64 box3 | 0.13ms | 0.32ms (grey_closing) | 0.39x | 4.2MB | 1.0MB | 4x |
| 64x64x64 ball2 | 6.03ms | 0.71ms (grey_closing) | 8.54x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 702.73ms | 3.29ms (grey_closing) | 213.44x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 1.19ms | 4.50ms (grey_closing) | 0.26x | 268.4MB | 67.1MB | 4x |
| 256x256x256 ball2 | 1394.04ms | 4.59ms (grey_closing) | 303.67x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 26.81ms (grey_closing) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 3/7 sizes; baseline ahead at `256x256x256 ball2` 303.67x, `4*128x128x128 ball2` 213.44x, `64x64x64 ball2` 8.54x, `256x256 disk3` 1.21x; at the largest `256x256x256 ball2`, baseline 303.67x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_closing ran (26.81ms).

## dilate  (nitrix.morphology.dilate)  [jax-cuda12]

**Cost law.** time: flat box O(N) (fused reduce_window) vs explicit SE O(N*k^d) (im2col); HBM: box O(N), explicit-SE im2col O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy/scipy (O(N*k), in-place) hold. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.09ms | 0.13ms (grey_dilation) | 0.72x | 0.8MB | 0.3MB | 3x |
| 256x256 box15 | 0.15ms | 0.14ms (grey_dilation) | 1.06x | 0.8MB | 0.3MB | 3x |
| 256x256 disk3 | 0.38ms | 0.29ms (grey_dilation) | 1.28x | 93.1MB | 0.3MB | 355x |
| 256x256 disk7 | 1.58ms | 0.31ms (grey_dilation) | 5.13x | 193.5MB | 0.3MB | 735x |
| 256x256 box3,float16 | 0.09ms | — | ok | 0.4MB | — | — |
| 64x64x64 box3 | 0.11ms | 0.18ms (grey_dilation) | 0.60x | 3.1MB | 1.0MB | 3x |
| 64x64x64 ball2 | 3.09ms | 0.34ms (grey_dilation) | 8.97x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 351.58ms | 1.87ms (grey_dilation) | 187.80x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 0.63ms | 2.29ms (grey_dilation) | 0.28x | 201.3MB | 67.1MB | 3x |
| 256x256x256 ball2 | 698.60ms | 2.35ms (grey_dilation) | 297.33x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 12.02ms (grey_dilation) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 3/9 sizes; baseline ahead at `256x256x256 ball2` 297.33x, `4*128x128x128 ball2` 187.80x, `64x64x64 ball2` 8.97x, `256x256 disk7` 5.13x (+2 more); at the largest `256x256x256 ball2`, baseline 297.33x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_dilation ran (12.02ms).

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

## erode  (nitrix.morphology.erode)  [jax-cuda12]

**Cost law.** time: flat box O(N) (fused reduce_window) vs explicit SE O(N*k^d) (im2col); HBM: box O(N), explicit-SE im2col O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy/scipy (O(N*k), in-place) hold. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.10ms | 0.12ms (grey_erosion) | 0.83x | 0.8MB | 0.3MB | 3x |
| 256x256 box15 | 0.10ms | 0.13ms (grey_erosion) | 0.73x | 0.8MB | 0.3MB | 3x |
| 256x256 disk3 | 0.38ms | 0.26ms (grey_erosion) | 1.49x | 93.1MB | 0.3MB | 355x |
| 256x256 disk7 | 1.58ms | 0.28ms (grey_erosion) | 5.74x | 193.5MB | 0.3MB | 735x |
| 256x256 box3,float16 | 0.10ms | — | ok | 0.4MB | — | — |
| 64x64x64 box3 | 0.10ms | 0.17ms (grey_erosion) | 0.57x | 3.1MB | 1.0MB | 3x |
| 64x64x64 ball2 | 3.09ms | 0.29ms (grey_erosion) | 10.68x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 351.55ms | 1.86ms (grey_erosion) | 189.23x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 0.63ms | 2.28ms (grey_erosion) | 0.28x | 201.3MB | 67.1MB | 3x |
| 256x256x256 ball2 | 698.45ms | 2.31ms (grey_erosion) | 302.77x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 11.81ms (grey_erosion) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 4/9 sizes; baseline ahead at `256x256x256 ball2` 302.77x, `4*128x128x128 ball2` 189.23x, `64x64x64 ball2` 10.68x, `256x256 disk7` 5.74x (+1 more); at the largest `256x256x256 ball2`, baseline 302.77x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_erosion ran (11.81ms).

## laplacian_eigenmap  (nitrix.graph.laplacian_eigenmap)  [jax-cuda12]

**Cost law.** dense O(n^3) eigh / O(n^2) backend+operator -> infeasible at n~100k (~40 GB dense); sparse lobpcg O(iters*nnz) fwd + O(nnz*k) differentiable backward -> scales (fsaverage6/7), and is the only differentiable option (scipy/cupy eigsh have no gradient).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=1024 dense | — | 31.32ms (eigsh) | skipped | — | 4.2MB | — |
| n=1024 dense k32 | — | 86.25ms (eigsh) | skipped | — | 4.2MB | — |
| n=2048 dense | — | 69.52ms (eigsh) | skipped | — | 16.8MB | — |
| n=2048 ell | 179.47ms | 50.75ms (eigsh) | 3.54x | 121.0MB | 20.2MB | 6x |
| n=4096 ell | 225.66ms | 83.16ms (eigsh) | 2.71x | 101.0MB | 80.3MB | 1x |
| n=10242 ell | 42.59ms | 609.07ms (eigsh) | 0.07x | 143.3MB | 8.4MB | 17x |
| n=40962 ell | 23.55ms | 1002.00ms (eigsh) | 0.02x | 153.9MB | 33.6MB | 5x |
| n=120000 ell | 47.35ms | 3273.97ms (eigsh) | 0.01x | 204.5MB | 67.1MB | 3x |

- **Speed:** nitrix wins 3/5 sizes; baseline ahead at `n=2048 ell` 3.54x, `n=4096 ell` 2.71x; at the largest `n=120000 ell`, nitrix 69.15x ahead.
- **Projected OOM (≈24GB):** nitrix ~225.4 Melem vs best baseline ~687 Melem (~3x more headroom).
- **Dispatch note (not a scale risk):** nitrix `skipped` at `n=1024 dense`, `n=1024 dense k32`, `n=2048 dense` (the default path is unavailable on this platform -- e.g. the cuSolver eigh block -- while the reference ran).

## open  (nitrix.morphology.open)  [jax-cuda12]

**Cost law.** time: flat box O(N) (two fused reduce_windows) vs explicit SE O(N*k^d) (two im2col passes); HBM: box O(N), explicit-SE O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy (O(N*k), in-place) holds. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.10ms | 0.22ms (grey_opening) | 0.45x | 1.0MB | 0.3MB | 4x |
| 256x256 disk3 | 0.64ms | 0.52ms (grey_opening) | 1.22x | 93.1MB | 0.3MB | 355x |
| 64x64x64 box3 | 0.12ms | 0.33ms (grey_opening) | 0.36x | 4.2MB | 1.0MB | 4x |
| 64x64x64 ball2 | 6.00ms | 0.56ms (grey_opening) | 10.65x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 702.79ms | 3.29ms (grey_opening) | 213.82x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 1.15ms | 4.50ms (grey_opening) | 0.26x | 268.4MB | 67.1MB | 4x |
| 256x256x256 ball2 | 1396.49ms | 4.49ms (grey_opening) | 310.74x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 27.08ms (grey_opening) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 3/7 sizes; baseline ahead at `256x256x256 ball2` 310.74x, `4*128x128x128 ball2` 213.82x, `64x64x64 ball2` 10.65x, `256x256 disk3` 1.22x; at the largest `256x256x256 ball2`, baseline 310.74x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_opening ran (27.08ms).

