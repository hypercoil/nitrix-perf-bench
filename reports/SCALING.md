# Scaling / crossover report

Scale-gaming defence: the scaling curve + the stated cost law, so a small-size win cannot hide a large-size / batched loss or OOM. Platform: `jax-cuda12`.

## affine_register  (nitrix.register.affine_register)  [jax-cuda12]

**Cost law.** post loop-roll (lax.scan): COMPILE ~flat in iterations, ~4-11 s (was 24-211 s unrolled; the L3x30 CPU compile that failed XLA now compiles). STEADY ~ iterations x P x N with P=12 (assemble J^TJ + a matrix_exp of the linear block + a P x P solve) -- ~2x rigid per-iter. GPU steady is overhead-bound below ~48^3 then compute-bound; the GPU/CPU speedup climbs to a brain-scale plateau ~35x. HBM like rigid (J is P-thin); cold peak_hbm is autotune-contaminated -- no OOM projection (see reports/REGISTRATION_SCALING.md). Bias: fixed (levels=2, iters=20); real pipelines raise levels with resolution.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 11.49ms | — | ok | 219.0MB | — | — |
| 96x96x96 | 26.36ms | — | ok | 1279.8MB | — | — |
| 96x96x96 world | 78.45ms | — | ok | 1484.8MB | — | — |
| mni152 2mm | 36.59ms | — | ok | 1634.8MB | — | — |
| 128x128x128 | 69.68ms | — | ok | 8733.6MB | — | — |
| 128x128x128 world | 198.12ms | — | ok | 8758.2MB | — | — |
| 160x160x160 | 138.91ms | — | ok | 13694.1MB | — | — |
| 192x192x192 | 244.58ms | — | ok | 1374.6MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~123.6 Melem.

## bbr_register  (nitrix.register.bbr_register)  [jax-cuda12]

**Cost law.** STEADY ~ iters x N: BFGS over the (rigid) parameters, each cost eval samples 2N points along the boundary normals + a tanh contrast -- VOLUME-INDEPENDENT (only 2N samples touch the grid). NO ITK/ANTs equivalent (a nitrix-only capability; the comparison is GPU vs CPU + the one-time compile, no domain tool). HBM ~ N (the point arrays), tiny. The size tier varies N to cortical-mesh scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| N2000 48x48x48 | 6.32ms | — | ok | 336.0MB | — | — |
| N5000 64x64x64 | 9.63ms | — | ok | 336.7MB | — | — |
| N20000 64x64x64 | 3.44ms | — | ok | 337.1MB | — | — |
| N80000 64x64x64 | 5.86ms | — | ok | 338.6MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~5.7 Melem.

## close  (nitrix.morphology.close)  [jax-cuda12]

**Cost law.** time: flat box O(N) (two fused reduce_windows) vs explicit SE O(N*k^d) (two im2col passes); HBM: box O(N), explicit-SE O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy (O(N*k), in-place) holds. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.09ms | 0.24ms (grey_closing) | 0.40x | 1.0MB | 0.3MB | — |
| 256x256 disk3 | 0.64ms | 0.53ms (grey_closing) | 1.21x | 93.1MB | 0.3MB | — |
| 64x64x64 box3 | 0.13ms | 0.32ms (grey_closing) | 0.39x | 4.2MB | 1.0MB | 4x |
| 64x64x64 ball2 | 6.03ms | 0.71ms (grey_closing) | 8.54x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 702.73ms | 3.29ms (grey_closing) | 213.44x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 1.19ms | 4.50ms (grey_closing) | 0.26x | 268.4MB | 67.1MB | 4x |
| 256x256x256 ball2 | 1394.04ms | 4.59ms (grey_closing) | 303.67x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 26.81ms (grey_closing) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 3/7 sizes; baseline ahead at `256x256x256 ball2` 303.67x, `4*128x128x128 ball2` 213.44x, `64x64x64 ball2` 8.54x, `256x256 disk3` 1.21x; at the largest `256x256x256 ball2`, baseline 303.67x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_closing ran (26.81ms).

## conditionalcorr  (nitrix.stats.conditionalcorr)  [jax-cuda12]

**Cost law.** residualise (OLS: a (d, d) Gram + Cholesky O(d^3) + projection O(c * obs * d)) then cov O(c^2 * obs) then a geometric-mean normalisation. The cov dominates at parcel c, so matmul-bound and GPU-robust -- the (d, d) solver is tiny (contrast pca_fit). HBM ~ c * obs (input) + c^2 (output). The size tier varies c to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c128 d16 obs1024 | 0.17ms | 0.48ms (conditionalcorr) | 0.37x | 73.4MB | 0.6MB | — |
| c256 d16 obs2048 | 0.22ms | 0.47ms (conditionalcorr) | 0.46x | 79.7MB | 2.2MB | 36x |
| c512 d16 obs4096 | 0.39ms | 0.55ms (conditionalcorr) | 0.72x | 100.7MB | 8.7MB | 12x |
| c1024 d16 obs4096 | 0.92ms | 1.30ms (conditionalcorr) | 0.71x | 123.0MB | 17.0MB | 7x |
| c2048 d32 obs8192 | 6.70ms | 9.11ms (conditionalcorr) | 0.74x | 370.1MB | 68.2MB | 5x |

- **Speed:** nitrix wins 5/5 sizes; at the largest `c2048 d32 obs8192`, nitrix 1.36x ahead.
- **Projected OOM (≈24GB):** nitrix ~1087.8 Melem vs best baseline ~5908 Melem (~5x more headroom).

## conditionalcov  (nitrix.stats.conditionalcov)  [jax-cuda12]

**Cost law.** residualise (OLS: a (d, d) Gram + Cholesky O(d^3) + projection O(c * obs * d)) then cov O(c^2 * obs). The cov dominates at parcel c (d is a handful of confounds), so the op is matmul-bound and GPU-robust -- the (d, d) solver is tiny (contrast pca_fit). HBM ~ c * obs (input) + c^2 (output). The size tier varies c to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c128 d16 obs1024 | 0.19ms | 0.41ms (conditionalcov) | 0.46x | 73.4MB | 0.6MB | — |
| c256 d16 obs2048 | 0.21ms | 0.43ms (conditionalcov) | 0.50x | 79.7MB | 2.2MB | 36x |
| c512 d16 obs4096 | 0.40ms | 0.53ms (conditionalcov) | 0.76x | 100.7MB | 8.7MB | 12x |
| c1024 d16 obs4096 | 0.91ms | 1.24ms (conditionalcov) | 0.73x | 123.0MB | 17.0MB | 7x |
| c2048 d32 obs8192 | 7.46ms | 8.86ms (conditionalcov) | 0.84x | 370.1MB | 68.2MB | 5x |

- **Speed:** nitrix wins 5/5 sizes; at the largest `c2048 d32 obs8192`, nitrix 1.19x ahead.
- **Projected OOM (≈24GB):** nitrix ~1087.8 Melem vs best baseline ~5908 Melem (~5x more headroom).

## corr  (nitrix.stats.corr)  [jax-cuda12]

**Cost law.** corrcoef = the centred/standardised cov: O(n^2 * t) -- a single BLAS-class matmul (same GPU-friendly regime as cov), the larger n being where it pulls ahead. HBM ~ n^2 (the n x n output). The size tier varies n to large-parcellation scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=50 | 0.11ms | 0.13ms (corrcoef) | 0.88x | 71.6MB | 0.1MB | — |
| n=500 | 0.23ms | 3.71ms (corrcoef) | 0.06x | 83.5MB | 4.2MB | 20x |
| n=2000 | 0.77ms | 21.64ms (corrcoef) | 0.04x | 111.7MB | 8.4MB | 13x |

- **Speed:** nitrix wins 3/3 sizes; at the largest `n=2000`, nitrix 27.95x ahead.
- **Projected OOM (≈24GB):** nitrix ~0.4 Melem vs best baseline ~6 Melem (~13x more headroom).

## cov  (nitrix.stats.cov)  [jax-cuda12]

**Cost law.** centred X @ X.T / (n-1): O(c^2 * n_obs) -- a single BLAS-class matmul, the GPU-friendly regime (the larger c is where the matmul path pulls ahead of the CPU floor). HBM ~ c^2 (c x c output). The size tier varies c to large-parcellation scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c=50 | 0.11ms | 0.11ms (cov) | 1.05x | 71.6MB | 0.1MB | — |
| c=500 | 0.22ms | 3.68ms (cov) | 0.06x | 83.5MB | 4.2MB | 20x |
| c=2000 | 0.73ms | 21.27ms (cov) | 0.03x | 111.7MB | 8.4MB | 13x |

- **Speed:** nitrix wins 2/3 sizes; baseline ahead at `c=50` 1.05x; at the largest `c=2000`, nitrix 29.09x ahead.
- **Projected OOM (≈24GB):** nitrix ~859.5 Melem vs best baseline ~11444 Melem (~13x more headroom).

## diffeomorphic_demons  (nitrix.register.diffeomorphic_demons_register)  [jax-cuda12]

**Cost law.** post loop-roll (lax.scan): COMPILE flat in iterations -- L2x20 == L2x40 (~6.8 s on the L4); even the default L3x80 (240 iters), once minutes unrolled, is ~7 s. STEADY ~ iterations x n_steps x N (ESM force + 2 spatial_gradients + n_steps scaling-squaring warps + 2 Gaussians; no inner solve), but SUPER-linear at large N (bandwidth-bound on the SVF field): the GPU/CPU speedup peaks ~43x (48-96^3) then erodes to ~28x (160^3) -- the most bandwidth-bound recipe at scale. HBM: the heaviest recipe (~3 vs rigid/affine ~1.8 KB/voxel at clean small sizes), but cold peak_hbm is contaminated by XLA autotune scratch (a shared ~8.7 GB 128^3 spike, non-monotonic) so NO OOM projection is trustworthy; none hit OOM to 160^3 on the 23 GB L4. See reports/REGISTRATION_SCALING.md.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 18.43ms | 373.49ms (demons) | 0.05x | 339.1MB | 0.9MB | — |
| 96x96x96 | 79.46ms | 4032.41ms (demons) | 0.02x | 3777.4MB | 8.4MB | 450x |
| 96x96x96 aniso1x1x3 | 84.16ms | 4194.85ms (demons) | 0.02x | 3777.4MB | 8.4MB | 450x |
| mni152 2mm | 117.75ms | 4188.95ms (demons) | 0.03x | 15019.3MB | 16.8MB | 895x |
| 128x128x128 | 295.31ms | 8629.84ms (demons) | 0.03x | 8783.4MB | 16.8MB | 524x |
| 128x128x128 aniso1x1x3 | 301.42ms | 9396.10ms (demons) | 0.03x | 8850.5MB | 16.8MB | 528x |
| 160x160x160 | 647.53ms | 19915.77ms (demons) | 0.03x | 2652.2MB | 33.6MB | 79x |

- **Speed:** nitrix wins 7/7 sizes; at the largest `160x160x160`, nitrix 30.76x ahead.
- **Projected OOM (≈24GB):** nitrix ~37.1 Melem vs best baseline ~2930 Melem (~79x more headroom).

## diffusion_embedding  (nitrix.graph.diffusion_embedding)  [jax-cuda12]

**Cost law.** dense O(n^3) eigh / O(n^2) operator -> infeasible at n~100k (~40 GB dense diffusion operator); sparse lobpcg O(iters*nnz) fwd + O(nnz*k) differentiable backward -> scales (fsaverage6/7), and is the only differentiable option (scipy/cupy eigsh have no gradient).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=1024 dense | — | 30.44ms (eigsh) | skipped | — | 4.2MB | — |
| n=2048 dense | — | 44.67ms (eigsh) | skipped | — | 16.8MB | — |
| n=2048 ell | 97.00ms | 44.71ms (eigsh) | 2.17x | 121.0MB | 20.2MB | 6x |
| n=10242 ell | 42.00ms | 129.82ms (eigsh) | 0.32x | 143.3MB | 8.4MB | 17x |
| n=40962 ell | 22.73ms | 202.86ms (eigsh) | 0.11x | 153.9MB | 33.6MB | 5x |
| n=120000 ell | 46.31ms | 374.97ms (eigsh) | 0.12x | 204.5MB | 67.1MB | 3x |

- **Speed:** nitrix wins 3/4 sizes; baseline ahead at `n=2048 ell` 2.17x; at the largest `n=120000 ell`, nitrix 8.10x ahead.
- **Projected OOM (≈24GB):** nitrix ~225.4 Melem vs best baseline ~687 Melem (~3x more headroom).
- **Dispatch note (not a scale risk):** nitrix `skipped` at `n=1024 dense`, `n=2048 dense` (the default path is unavailable on this platform -- e.g. the cuSolver eigh block -- while the reference ran).

## dilate  (nitrix.morphology.dilate)  [jax-cuda12]

**Cost law.** time: flat box O(N) (fused reduce_window) vs explicit SE O(N*k^d) (im2col); HBM: box O(N), explicit-SE im2col O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy/scipy (O(N*k), in-place) hold. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.09ms | 0.13ms (grey_dilation) | 0.72x | 0.8MB | 0.3MB | — |
| 256x256 box15 | 0.15ms | 0.14ms (grey_dilation) | 1.06x | 0.8MB | 0.3MB | — |
| 256x256 disk3 | 0.38ms | 0.29ms (grey_dilation) | 1.28x | 93.1MB | 0.3MB | — |
| 256x256 disk7 | 1.58ms | 0.31ms (grey_dilation) | 5.13x | 193.5MB | 0.3MB | — |
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
| 64x64 | 0.15ms | 0.20ms (distance_transform_edt) | 0.75x | 33.6MB | 0.0MB | — |
| 128x128 | 0.22ms | 0.20ms (distance_transform_edt) | 1.09x | 33.8MB | 0.1MB | — |
| 256x256 | 0.34ms | 0.24ms (distance_transform_edt) | 1.40x | 34.3MB | 0.3MB | — |
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
| 256x256 box3 | 0.10ms | 0.12ms (grey_erosion) | 0.83x | 0.8MB | 0.3MB | — |
| 256x256 box15 | 0.10ms | 0.13ms (grey_erosion) | 0.73x | 0.8MB | 0.3MB | — |
| 256x256 disk3 | 0.38ms | 0.26ms (grey_erosion) | 1.49x | 93.1MB | 0.3MB | — |
| 256x256 disk7 | 1.58ms | 0.28ms (grey_erosion) | 5.74x | 193.5MB | 0.3MB | — |
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

## flame_two_level  (nitrix.stats.lme.flame_two_level)  [jax-cuda12]

**Cost law.** batched single-param REML for the between-subject variance over V voxels (FSL FLAME equiv): O(V * iters * N) -- linear in the voxel batch V. MEASURED (L4): scales cleanly on the GPU through the dev tier to V=65536, but V>=131072 fails the GPU SOLVER (gpusolverDnCreate -- a cuSOLVER-class blocker on this box), a hard GPU scale CEILING; the batched FLAME still runs on CPU at brain-volume V. No fair external perf competitor (statsmodels cannot consume known per-subject variances; FSL FLAME is file-coupled). HBM ~ V.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| V=1024 | — | — | skipped | — | — | — |
| V=8192 | — | — | skipped | — | — | — |
| V=65536 | — | — | skipped | — | — | — |


## greedy_syn_register  (nitrix.register.greedy_syn_register)  [jax-cuda12]

**Cost law.** STEADY ~ levels x iters x n_steps x N: each iteration warps both images to the midpoint (two scaling-and-squaring SVF integrations), computes the LNCC force, smooths it (fluid) + the velocity (diffusion) -- two Gaussians/iter -- then a midpoint compose+invert at the end. The heaviest recipe to COMPILE (two velocity fields), but ANTs SyNOnly (the gold standard) is FAST on CPU (~0.5/2.9/6.0 s at 48/96/128^3 measured), so the GPU win is NOT a given -- it must clear the ~4x cost bar to count (measured in ECONOMIC.md, not assumed). HBM ~ 2 velocity fields + scaling-squaring intermediates (heaviest after demons). The size tier varies the volume + carries anisotropic (1x1x3) points.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 184.43ms | — | ok | 339.1MB | — | — |
| 64x64x64 | 178.89ms | — | ok | 1711.4MB | — | — |
| 64x64x64 aniso1x1x3 | 205.60ms | — | ok | 1711.4MB | — | — |
| 96x96x96 | 741.84ms | — | ok | 3777.4MB | — | — |
| 96x96x96 aniso1x1x3 | 775.93ms | — | ok | 3777.4MB | — | — |
| mni152 2mm | 1062.44ms | — | ok | 4749.4MB | — | — |
| 128x128x128 | 2459.94ms | — | ok | 8783.4MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~5.7 Melem.

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

## matrix_exp  (nitrix.linalg.matrix_exp)  [jax-cuda12]

**Cost law.** both O(n^3): nitrix is a ~(taylor_order + n_squarings) ~= 20 matmul stack (no factorisation); scipy/jax expm is Padé + a dense LU solve. Measured (this L4): nitrix 1.4-3.4x faster than jax expm on GPU (the saved solve; margin narrows as the shared O(n^3) matmul dominates at n=1024), 21-73x vs scipy CPU at n>=256 but slower at n=16 (launch overhead). HBM O(n^2) (a few n x n temporaries), flat ~90-105 MB.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=16 | 0.16ms | 0.03ms (expm) | 6.31x | 88.1MB | 0.0MB | — |
| n=64 | 0.19ms | 0.17ms (expm) | 1.11x | 88.1MB | 0.0MB | — |
| n=256 | 0.27ms | 0.93ms (expm) | 0.29x | 89.1MB | 72.1MB | 1x |
| n=512 | 0.67ms | 1.91ms (expm) | 0.35x | 92.3MB | 74.4MB | 1x |
| n=1024 | 3.42ms | 4.76ms (expm) | 0.72x | 104.9MB | 83.9MB | 1x |

- **Speed:** nitrix wins 3/5 sizes; baseline ahead at `n=16` 6.31x, `n=64` 1.11x; at the largest `n=1024`, nitrix 1.39x ahead.
- **Projected OOM (≈24GB):** nitrix ~0.2 Melem vs best baseline ~0 Melem (~1x more headroom).

## open  (nitrix.morphology.open)  [jax-cuda12]

**Cost law.** time: flat box O(N) (two fused reduce_windows) vs explicit SE O(N*k^d) (two im2col passes); HBM: box O(N), explicit-SE O(N*k^d) -> 256^3 ball OOMs (~49 GB) while cupy (O(N*k), in-place) holds. The flat box scales; the disk/ball footprint does not.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 256x256 box3 | 0.10ms | 0.22ms (grey_opening) | 0.45x | 1.0MB | 0.3MB | — |
| 256x256 disk3 | 0.64ms | 0.52ms (grey_opening) | 1.22x | 93.1MB | 0.3MB | — |
| 64x64x64 box3 | 0.12ms | 0.33ms (grey_opening) | 0.36x | 4.2MB | 1.0MB | 4x |
| 64x64x64 ball2 | 6.00ms | 0.56ms (grey_opening) | 10.65x | 336.6MB | 1.0MB | 321x |
| 4*128x128x128 ball2 | 702.79ms | 3.29ms (grey_opening) | 213.82x | 8724.2MB | 33.6MB | 260x |
| 256x256x256 box3 | 1.15ms | 4.50ms (grey_opening) | 0.26x | 268.4MB | 67.1MB | 4x |
| 256x256x256 ball2 | 1396.49ms | 4.49ms (grey_opening) | 310.74x | 16995.3MB | 67.1MB | 253x |
| 256x256x256 ball4 | — | 27.08ms (grey_opening) | oom | — | 67.1MB | — |

- **Speed:** nitrix wins 3/7 sizes; baseline ahead at `256x256x256 ball2` 310.74x, `4*128x128x128 ball2` 213.82x, `64x64x64 ball2` 10.65x, `256x256 disk3` 1.22x; at the largest `256x256x256 ball2`, baseline 310.74x ahead.
- **Projected OOM (≈24GB):** nitrix ~23.7 Melem vs best baseline ~6000 Melem (~253x more headroom).
- **OOM-as-signal:** nitrix `oom` at `256x256x256 ball4` while grey_opening ran (27.08ms).

## pairedcorr  (nitrix.stats.pairedcorr)  [jax-cuda12]

**Cost law.** cross-cov O(c * d * obs) then a geometric-mean normalisation. nitrix forms the full cov(X) O(c^2 * obs) and cov(Y) O(d^2 * obs) to take their diagonals -- ~3x the minimal matmul (same complexity class); the floor computes the variances directly. Solver-free (GPU). HBM ~ (c + d) * obs. The size tier varies c = d to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c128 d128 obs1024 | 0.13ms | 0.35ms (pairedcorr) | 0.36x | 86.1MB | 1.0MB | 82x |
| c256 d256 obs2048 | 0.21ms | 0.37ms (pairedcorr) | 0.57x | 88.6MB | 4.2MB | 21x |
| c512 d512 obs4096 | 0.81ms | 0.38ms (pairedcorr) | 2.11x | 107.0MB | 16.8MB | 6x |
| c1024 d1024 obs4096 | 2.37ms | 1.18ms (pairedcorr) | 2.01x | 176.2MB | 33.6MB | 5x |
| c2048 d2048 obs8192 | 19.16ms | 9.21ms (pairedcorr) | 2.08x | 436.2MB | 134.2MB | 3x |

- **Speed:** nitrix wins 2/5 sizes; baseline ahead at `c512 d512 obs4096` 2.11x, `c2048 d2048 obs8192` 2.08x, `c1024 d1024 obs4096` 2.01x; at the largest `c2048 d2048 obs8192`, baseline 2.08x ahead.
- **Projected OOM (≈24GB):** nitrix ~923.1 Melem vs best baseline ~3000 Melem (~3x more headroom).

## pairedcov  (nitrix.stats.pairedcov)  [jax-cuda12]

**Cost law.** Xc @ Yc^T / (obs - 1): O(c * d * obs) -- one BLAS-class matmul (plus O((c + d) * obs) centring), the GPU-friendly regime (no solver, contrast the precision / pca_fit families). HBM ~ (c + d) * obs (the inputs) + c * d (the cross-block). The size tier varies c = d to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c128 d128 obs1024 | 0.14ms | 0.19ms (pairedcov) | 0.73x | 73.4MB | 1.0MB | 70x |
| c256 d256 obs2048 | 0.14ms | 0.19ms (pairedcov) | 0.72x | 79.7MB | 4.2MB | 19x |
| c512 d512 obs4096 | 0.33ms | 0.33ms (pairedcov) | 1.03x | 96.5MB | 16.8MB | 6x |
| c1024 d1024 obs4096 | 0.93ms | 0.98ms (pairedcov) | 0.96x | 113.2MB | 33.6MB | 3x |
| c2048 d2048 obs8192 | 5.98ms | 7.32ms (pairedcov) | 0.82x | 436.2MB | 134.2MB | 3x |

- **Speed:** nitrix wins 4/5 sizes; baseline ahead at `c512 d512 obs4096` 1.03x; at the largest `c2048 d2048 obs8192`, nitrix 1.23x ahead.
- **Projected OOM (≈24GB):** nitrix ~923.1 Melem vs best baseline ~3000 Melem (~3x more headroom).

## partialcorr  (nitrix.stats.partialcorr)  [jax-cuda12]

**Cost law.** precision (cov O(c^2*obs) + inverse O(c^3)) then normalising by the geometric mean of the diagonal -- the inverse dominates at brain-parcel c; HBM ~ c^2. Same GPU inverse as precision (a measured scale-WIN: nitrix consumed-inv beats the cupy GPU inv increasingly with c). The size tier varies c to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c=128 | 0.35ms | 0.73ms (partialcorr) | 0.48x | 72.9MB | 0.5MB | — |
| c=256 | 0.59ms | 1.38ms (partialcorr) | 0.43x | 77.6MB | 2.1MB | 37x |
| c=512 | 1.60ms | 6.81ms (partialcorr) | 0.24x | 88.1MB | 8.4MB | 11x |

- **Speed:** nitrix wins 3/3 sizes; at the largest `c=512`, nitrix 4.25x ahead.
- **Projected OOM (≈24GB):** nitrix ~71.4 Melem vs best baseline ~750 Melem (~11x more headroom).

## partialcov  (nitrix.stats.partialcov)  [jax-cuda12]

**Cost law.** precision (cov O(c^2*obs) + inverse O(c^3)) then a sign flip on the off-diagonals -- the inverse dominates at brain-parcel c; HBM ~ c^2. Same GPU inverse as precision (a measured scale-WIN: nitrix consumed-inv beats the cupy GPU inv increasingly with c). The size tier varies c to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c=128 | 0.33ms | 0.72ms (partialcov) | 0.46x | 72.9MB | 0.5MB | — |
| c=256 | 0.60ms | 1.37ms (partialcov) | 0.44x | 77.6MB | 2.1MB | 37x |
| c=512 | 1.58ms | 6.80ms (partialcov) | 0.23x | 88.1MB | 8.4MB | 11x |

- **Speed:** nitrix wins 3/3 sizes; at the largest `c=512`, nitrix 4.29x ahead.
- **Projected OOM (≈24GB):** nitrix ~71.4 Melem vs best baseline ~750 Melem (~11x more headroom).

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

## precision  (nitrix.stats.precision)  [jax-cuda12]

**Cost law.** cov is O(c^2 * obs); the INVERSE is O(c^3) and dominates at brain-parcel c. HBM ~ c^2. MEASURED (L4): nitrix jits a consumed-inv that scales WELL on the GPU -- it beats the cupy GPU inverse-covariance by a GROWING margin (2.35x at c=256 -> 11.5x at c=2048), and cupy ran across the range (no cuSOLVER failure observed up to c=2048). numpy/nilearn are the CPU floor (slow at c>=1024). A scale-WIN; the size tier varies c to parcel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| c=128 | 0.35ms | 0.71ms (inv_cov) | 0.50x | 72.9MB | 0.5MB | — |
| c=256 | 0.61ms | 1.37ms (inv_cov) | 0.45x | 77.6MB | 2.1MB | 37x |
| c=512 | 1.60ms | 6.79ms (inv_cov) | 0.24x | 88.1MB | 8.4MB | 11x |

- **Speed:** nitrix wins 3/3 sizes; at the largest `c=512`, nitrix 4.25x ahead.
- **Projected OOM (≈24GB):** nitrix ~71.4 Melem vs best baseline ~750 Melem (~11x more headroom).

## reml_fit  (nitrix.stats.lme.reml_fit)  [jax-cuda12]

**Cost law.** batched variance-components REML (FaST-LMM spectral trick) over V voxels: O(V*(n^3 eig + iters*n)) -- linear in the voxel batch V, the scale axis. nitrix fits all V in ONE call; statsmodels LOOPS one iterative fit per voxel (~14 ms/voxel), so the batched-vs-looped speedup GROWS with V (it is the headline, and why statsmodels is a slow_baseline at scale). HBM ~ V. The size tier varies V to brain-voxel scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| V=64 | 11.33ms | — | ok | 151.1MB | — | — |
| V=256 | 17.09ms | — | ok | 151.2MB | — | — |
| V=1024 | 11.36ms | — | ok | 135.0MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~0.2 Melem.

## rigid_register  (nitrix.register.rigid_register)  [jax-cuda12]

**Cost law.** post loop-roll (lax.scan): COMPILE ~flat in iterations AND volume (XLA compiles the per-iteration op graph, not an unrolled chain) -- ~4-11 s across configs/sizes (was 16-211 s unrolled). STEADY is the headline ~ iterations x P x N: each LM iter assembles the small-P normal equations J^TJ (P=6; ~P forward warp-passes + a P x P solve). GPU steady is overhead-bound (~flat) below ~48^3 then compute-bound (~N); the GPU/CPU speedup climbs from ~4x (24^3) to a brain-scale plateau ~25x. HBM: lighter than demons, but cold peak_hbm is autotune-contaminated at large N -- no OOM projection (see reports/REGISTRATION_SCALING.md). Bias: the size tier fixes (levels=2, iters=20); real pipelines raise levels with resolution.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 5.47ms | — | ok | 203.4MB | — | — |
| 96x96x96 | 20.24ms | — | ok | 1279.8MB | — | — |
| 96x96x96 world | 49.24ms | — | ok | 1514.6MB | — | — |
| mni152 2mm | 29.22ms | — | ok | 1606.0MB | — | — |
| 128x128x128 | 61.33ms | — | ok | 8733.6MB | — | — |
| 128x128x128 world | 137.27ms | — | ok | 8758.2MB | — | — |
| 160x160x160 | 125.17ms | — | ok | 13694.1MB | — | — |
| 192x192x192 | 225.81ms | — | ok | 1366.6MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~124.3 Melem.

## volreg  (nitrix.register.volreg)  [jax-cuda12]

**Cost law.** STEADY ~ T x iters x N per-frame, but the reference work (pyramid, inverse-compositional steepest-descent + Hessian) is hoisted once and the T frames are vmap-batched behind ONE compile -- so nitrix-GPU stays sublinear in T once the batch fills the device, while ANTs is T sequential CPU registrations (~T x 60 ms). The GPU:CPU gap should GROW with T (the batching/amortisation story); the honest CPU bar is the FAST community tool (3dvolreg / mcflirt), I/O-floor-subtracted, not the slower ANTs (timed out at T=500). HBM ~ T*N (realigned series + vmap working set) -- the binding constraint; OOM at the top is reported as signal. Size tier varies T (headline) + volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| T8 32x32x32 | 1.02ms | — | ok | 135.5MB | — | — |
| T16 32x32x32 | 1.35ms | — | ok | 136.3MB | — | — |
| T32 32x32x32 | 3.02ms | — | ok | 138.4MB | — | — |
| T50 48x48x48 | 81.24ms | — | ok | 334.8MB | — | — |
| T100 48x48x48 | 172.76ms | — | ok | 625.3MB | — | — |
| T200 48x48x48 | 355.58ms | — | ok | 1208.0MB | — | — |
| T100 64x64x64 | 443.45ms | — | ok | 1347.5MB | — | — |
| T100 80x80x80 | 1073.26ms | — | ok | 2638.0MB | — | — |
| T500 48x48x48 | 904.76ms | — | ok | 2815.2MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~471.4 Melem.

