# Scaling / crossover report

Scale-gaming defence: the scaling curve + the stated cost law, so a small-size win cannot hide a large-size / batched loss or OOM. Platform: `jax-cuda12`.

## affine_exp  (nitrix.geometry.affine_exp)  [jax-cuda12]

**Cost law.** O(B) over the batch B, embarrassingly parallel, but heavier per element than rigid_exp: a matrix_exp (scaling-and-squaring, ~20 3x3 matmuls) of the gl(3) generator + a direct translation. Throughput-bound; HBM ~ B. The batch tier varies B. (The 4x4 matrix is the small-N regime of the matrix_exp case, here batched.)

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| b=1024 | 0.15ms | 1.87ms (affine_exp) | 0.08x | 0.7MB | 0.0MB | — |
| b=16384 | 0.18ms | 1.86ms (affine_exp) | 0.10x | 11.5MB | 0.8MB | — |
| b=65536 | 0.32ms | 3.99ms (affine_exp) | 0.08x | 46.1MB | 4.2MB | 11x |
| b=262144 | 1.43ms | 15.80ms (affine_exp) | 0.09x | 184.5MB | 16.8MB | 11x |
| b=1048576 | 13.54ms | 99.29ms (affine_exp) | 0.14x | 738.2MB | 67.1MB | 11x |

- **Speed:** nitrix wins 5/5 sizes; at the largest `b=1048576`, nitrix 7.33x ahead.
- **Projected OOM (≈24GB):** nitrix ~34.1 Melem vs best baseline ~375 Melem (~11x more headroom).

## affine_register  (nitrix.register.affine_register)  [jax-cuda12]

**Cost law.** post loop-roll (lax.scan): COMPILE ~flat in iterations, ~4-11 s (was 24-211 s unrolled; the L3x30 CPU compile that failed XLA now compiles). STEADY ~ iterations x P x N with P=12 (assemble J^TJ + a matrix_exp of the linear block + a P x P solve) -- ~2x rigid per-iter. GPU steady is overhead-bound below ~48^3 then compute-bound; the GPU/CPU speedup climbs to a brain-scale plateau ~35x. HBM like rigid (J is P-thin); cold peak_hbm is autotune-contaminated -- no OOM projection (see reports/REGISTRATION_SCALING.md). Bias: fixed (levels=2, iters=20); real pipelines raise levels with resolution.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 12.83ms | — | ok | 470.7MB | — | — |
| 96x96x96 | 35.62ms | — | ok | 1279.6MB | — | — |
| 96x96x96 world | 73.22ms | — | ok | 593.6MB | — | — |
| mni152 2mm | 27.49ms | — | ok | 1634.8MB | — | — |
| 128x128x128 | 52.16ms | — | ok | 8733.2MB | — | — |
| 128x128x128 world | 176.37ms | — | ok | 734.0MB | — | — |
| 160x160x160 | 119.53ms | — | ok | 13643.8MB | — | — |
| 192x192x192 | 223.97ms | — | ok | 1374.6MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~123.6 Melem.

## bbr_register  (nitrix.register.bbr_register)  [jax-cuda12]

**Cost law.** STEADY ~ iters x N: BFGS over the (rigid) parameters, each cost eval samples 2N points along the boundary normals + a tanh contrast -- VOLUME-INDEPENDENT (only 2N samples touch the grid). NO ITK/ANTs equivalent (a nitrix-only capability; the comparison is GPU vs CPU + the one-time compile, no domain tool). HBM ~ N (the point arrays), tiny. The size tier varies N to cortical-mesh scale.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| N2000 48x48x48 | 6.21ms | — | ok | 336.0MB | — | — |
| N5000 64x64x64 | 1.41ms | — | ok | 336.7MB | — | — |
| N20000 64x64x64 | 5.28ms | — | ok | 337.1MB | — | — |
| N80000 64x64x64 | 10.14ms | — | ok | 338.6MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~5.7 Melem.

## bending_energy  (nitrix.register.bending_energy)  [jax-cuda12]

**Cost law.** O(N) over the voxel count N but ~2x gradient_smoothness: the displacement Jacobian, then a SECOND central-diff of each of its d*d components (the per-voxel Hessian, ~d*d*d stencil passes) + a Frobenius reduction. Stencil-heavy, bandwidth-bound, GPU-pure; HBM ~ N with a larger constant (the (d*d, d) Hessian intermediate materialises). The brain-scale tier varies the volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.32ms | 6.91ms (bending_energy) | 0.05x | 134.9MB | 0.4MB | — |
| 48^3 | 0.36ms | 7.24ms (bending_energy) | 0.05x | 204.3MB | 2.1MB | 97x |
| 64^3 | 0.57ms | 7.74ms (bending_energy) | 0.07x | 618.8MB | 4.2MB | 148x |
| 96^3 | 2.59ms | 13.84ms (bending_energy) | 0.19x | 1317.9MB | 16.8MB | 79x |
| 128^3 | 7.29ms | 46.39ms (bending_energy) | 0.16x | 8758.2MB | 33.6MB | 261x |
| 160^3 | 15.21ms | 99.28ms (bending_energy) | 0.15x | 13677.8MB | 67.1MB | 204x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 6.53x ahead.
- **Projected OOM (≈24GB):** nitrix ~7.2 Melem vs best baseline ~1465 Melem (~204x more headroom).

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

## compose_velocity  (nitrix.geometry.compose_velocity)  [jax-cuda12]

**Cost law.** O(N) over the voxel count N (order=2): two displacement Jacobians (central-diff stencils) + two ...ij,...j contractions + the add. Stencil + contraction, memory-bandwidth-bound, GPU-pure; HBM ~ N (the Jacobian intermediates). order=1 is a trivial elementwise add. The brain-scale tier varies the volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.12ms | 1.80ms (compose_velocity) | 0.07x | 103.0MB | 0.8MB | — |
| 48^3 | 0.12ms | 2.08ms (compose_velocity) | 0.06x | 109.6MB | 3.4MB | 32x |
| 64^3 | 0.13ms | 2.65ms (compose_velocity) | 0.05x | 121.6MB | 8.4MB | 14x |
| 96^3 | 0.26ms | 6.72ms (compose_velocity) | 0.04x | 182.8MB | 33.6MB | 5x |
| 128^3 | 1.06ms | 21.22ms (compose_velocity) | 0.05x | 318.8MB | 67.1MB | 5x |
| 160^3 | 1.98ms | 43.95ms (compose_velocity) | 0.05x | 536.9MB | 134.2MB | 4x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 22.15x ahead.
- **Projected OOM (≈24GB):** nitrix ~183.1 Melem vs best baseline ~732 Melem (~4x more headroom).

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

## connected_components  (nitrix.morphology.connected_components)  [jax-cuda12]

**Cost law.** jit-able label propagation with POINTER JUMPING (lax.while_loop: a neighbour-max hop + an L=L[L-1] pointer-jump per pass), O(log d) passes for diameter d, each O(N). MEASURED (L4): nitrix steady GROWS STEEPLY (1.3 -> 15.7 ms over 48 -> 160^3, ~12x) while cupyx label stays ~flat (0.6 -> 0.85 ms), so cupyx pulls from ~2x to ~18x ahead -- nitrix SCALES POORLY here, a kernel/algorithm scale risk (filed on nitrix main).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.68ms | 0.39ms (label) | 1.76x | 33.6MB | 0.0MB | — |
| 48^3 | 1.33ms | 0.40ms (label) | 3.35x | 33.7MB | 0.1MB | — |
| 64^3 | 1.64ms | 0.50ms (label) | 3.29x | 33.8MB | 0.3MB | — |
| 96^3 | 1.75ms | 0.52ms (label) | 3.39x | 34.4MB | 0.9MB | — |
| 128^3 | 6.06ms | 0.65ms (label) | 9.34x | 52.4MB | 2.1MB | 25x |
| 160^3 | 15.82ms | 0.87ms (label) | 18.23x | 104.9MB | 4.2MB | 25x |

- **Speed:** nitrix wins 0/6 sizes; baseline ahead at `160^3` 18.23x, `128^3` 9.34x, `96^3` 3.39x, `48^3` 3.35x (+2 more); at the largest `160^3`, baseline 18.23x ahead.
- **Projected OOM (≈24GB):** nitrix ~937.5 Melem vs best baseline ~23438 Melem (~25x more headroom).

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
| 48x48x48 | 12.63ms | 1170.55ms (demons) | 0.01x | 203.4MB | 0.9MB | — |
| 96x96x96 | 66.02ms | 4380.52ms (demons) | 0.02x | 1309.5MB | 8.4MB | 156x |
| 96x96x96 aniso1x1x3 | 51.29ms | 8041.49ms (demons) | 0.01x | 1309.5MB | 8.4MB | 156x |
| mni152 2mm | 69.31ms | 3126.70ms (demons) | 0.02x | 1634.8MB | 16.8MB | 97x |
| 128x128x128 | 281.27ms | 10566.57ms (demons) | 0.03x | 8758.2MB | 16.8MB | 522x |
| 128x128x128 aniso1x1x3 | 183.47ms | 10548.59ms (demons) | 0.02x | 8758.2MB | 16.8MB | 522x |
| 160x160x160 | 741.09ms | 19498.59ms (demons) | 0.04x | 13643.8MB | 33.6MB | 407x |

- **Speed:** nitrix wins 7/7 sizes; at the largest `160x160x160`, nitrix 26.31x ahead.
- **Projected OOM (≈24GB):** nitrix ~7.2 Melem vs best baseline ~2930 Melem (~407x more headroom).

## diffusion_embedding  (nitrix.graph.diffusion_embedding)  [jax-cuda12]

**Cost law.** dense O(n^3) eigh / O(n^2) operator -> infeasible at n~100k (~40 GB dense diffusion operator); sparse lobpcg O(iters*nnz) fwd + O(nnz*k) differentiable backward -> scales (fsaverage6/7), and is the only differentiable option (scipy/cupy eigsh have no gradient).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=1024 dense | — | 30.36ms (eigsh) | skipped | — | 4.2MB | — |
| n=2048 dense | — | 46.26ms (eigsh) | skipped | — | 16.8MB | — |
| n=2048 ell | 168.55ms | 45.02ms (eigsh) | 3.74x | 121.0MB | 20.2MB | 6x |
| n=10242 ell | 42.00ms | 129.82ms (eigsh) | 0.32x | 143.3MB | 8.4MB | 17x |
| n=40962 ell | 22.73ms | 202.86ms (eigsh) | 0.11x | 153.9MB | 33.6MB | 5x |
| n=120000 ell | 46.31ms | 374.97ms (eigsh) | 0.12x | 204.5MB | 67.1MB | 3x |

- **Speed:** nitrix wins 3/4 sizes; baseline ahead at `n=2048 ell` 3.74x; at the largest `n=120000 ell`, nitrix 8.10x ahead.
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

## distance_transform_edt  (nitrix.morphology.distance_transform_edt)  [jax-cuda12]

**Cost law.** separable min-plus SEMIRING EDT (all-parabola search, the euclidean alias of distance_transform): high-FLOP but shallow/parallel, so it WINS small and LOSES large vs F-H. MEASURED (L4): nitrix 2.4x ahead of cupyx at 48^3, crossing over ~96^3 to 2.2x behind at 160^3 (0.15->1.99 vs cupyx 0.36->0.89 ms) -- the known semiring trade-off. A scale-aware dispatch (semiring small, F-H large) keeps the win at both ends (filed lower-priority on nitrix main).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.14ms | 0.13ms (distance_transform_edt) | 1.10x | 33.8MB | 0.0MB | — |
| 48^3 | 0.18ms | 0.48ms (distance_transform_edt) | 0.37x | 34.5MB | 0.1MB | — |
| 64^3 | 0.28ms | 0.21ms (distance_transform_edt) | 1.35x | 36.7MB | 0.3MB | — |
| 96^3 | 0.37ms | 0.32ms (distance_transform_edt) | 1.17x | 41.5MB | 0.9MB | — |
| 128^3 | 0.48ms | 0.51ms (distance_transform_edt) | 0.93x | 52.4MB | 2.1MB | 25x |
| 160^3 | 1.97ms | 0.89ms (distance_transform_edt) | 2.22x | 71.3MB | 4.2MB | 17x |

- **Speed:** nitrix wins 2/6 sizes; baseline ahead at `160^3` 2.22x, `64^3` 1.35x, `96^3` 1.17x, `32^3` 1.10x; at the largest `160^3`, baseline 2.22x ahead.
- **Projected OOM (≈24GB):** nitrix ~1378.7 Melem vs best baseline ~23438 Melem (~17x more headroom).

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

**Cost law.** batched single-param REML for the between-subject variance over V voxels: O(V * iters * N) -- linear in the voxel batch V. nitrix fits all V in ONE call; FSL FLAME (flameo) and statsmodels meta-analysis LOOP one fit per voxel, so the batched-vs-looped speedup GROWS with V (the headline, and why both are slow_baselines). MEASURED (L4): the GPU solver path hits the cuSOLVER gpusolverDnCreate blocker at ALL V (gpu_solver_unavailable -- a graceful skip, seen in every stored run), so nitrix runs CPU-only here; even so the batched CPU fit beats the looped CPU tools (~4x flameo at V=1024). HBM ~ V.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| V=1024 | — | — | skipped | — | — | — |
| V=8192 | — | — | skipped | — | — | — |
| V=65536 | — | — | skipped | — | — | — |
| V=131072 | — | — | skipped | — | — | — |
| V=262144 | — | — | skipped | — | — | — |


## gradient_smoothness  (nitrix.register.gradient_smoothness)  [jax-cuda12]

**Cost law.** O(N) over the voxel count N: one roll-based central-diff pass (the displacement Jacobian) + a Frobenius reduction. Stencil + reduction, memory-bandwidth-bound and GPU-pure (no solver); HBM ~ N (a few d-component field copies). The brain-scale tier varies the volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.10ms | 0.88ms (gradient_smoothness) | 0.12x | 33.9MB | 0.4MB | — |
| 48^3 | 0.11ms | 1.04ms (gradient_smoothness) | 0.10x | 35.7MB | 2.1MB | 17x |
| 64^3 | 0.11ms | 1.25ms (gradient_smoothness) | 0.09x | 37.7MB | 4.2MB | 9x |
| 96^3 | 0.16ms | 3.25ms (gradient_smoothness) | 0.05x | 50.3MB | 16.8MB | 3x |
| 128^3 | 0.23ms | 11.27ms (gradient_smoothness) | 0.02x | 67.1MB | 33.6MB | 2x |
| 160^3 | 0.35ms | 22.91ms (gradient_smoothness) | 0.02x | 83.9MB | 67.1MB | 1x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 65.36x ahead.
- **Projected OOM (≈24GB):** nitrix ~1171.6 Melem vs best baseline ~1465 Melem (~1x more headroom).

## greedy_syn_register  (nitrix.register.greedy_syn_register)  [jax-cuda12]

**Cost law.** STEADY ~ levels x iters x n_steps x N: each iteration warps both images to the midpoint (two scaling-and-squaring SVF integrations), computes the LNCC force, smooths it (fluid) + the velocity (diffusion) -- two Gaussians/iter -- then a midpoint compose+invert at the end. The heaviest recipe to COMPILE (two velocity fields), but ANTs SyNOnly (the gold standard) is FAST on CPU (~0.5/2.9/6.0 s at 48/96/128^3 measured), so the GPU win is NOT a given -- it must clear the ~4x cost bar to count (measured in ECONOMIC.md, not assumed). HBM ~ 2 velocity fields + scaling-squaring intermediates (heaviest after demons). The size tier varies the volume + carries anisotropic (1x1x3) points. The force is a benchmarked knob: nitrix-jax (LNCC) vs the MI force (closed-form MIForce + autodiff MetricForce(MI)) -- MI replaces the local-CC window with a joint-histogram scatter (modality-independent cost; fMRIPrep parity, ANTs SyNOnly = mattes MI is the bar).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 74.60ms | — | ok | 291.8MB | — | — |
| 64x64x64 | 115.54ms | — | ok | 616.7MB | — | — |
| 64x64x64 aniso1x1x3 | 120.23ms | — | ok | 616.7MB | — | — |
| 96x96x96 | 398.74ms | — | ok | 1309.5MB | — | — |
| 96x96x96 aniso1x1x3 | 418.95ms | — | ok | 1309.5MB | — | — |
| mni152 2mm | 534.62ms | — | ok | 1634.8MB | — | — |
| 128x128x128 | 1236.47ms | — | ok | 8758.2MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~5.7 Melem.

## invert_displacement  (nitrix.geometry.invert_displacement)  [jax-cuda12]

**Cost law.** O(K x N): K Picard iterations (data-adaptive -- a lax.while_loop to relative tol, NOT a fixed scan), each a 3-channel linear-interp warp (gather) over N voxels. Compile flat (one while_loop body); steady ~ K x N, K set by convergence (‖∇s‖ controls K). The headline is capability: IFT-differentiable (numpy/cupy are not) + early-exit coexisting with the implicit backward. The brain-scale tier varies N (capped at 128^3 -- iterative + the cupy host-sync ref).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.71ms | 10.56ms (invert_displacement) | 0.07x | 33.9MB | 0.4MB | — |
| 48^3 | 1.11ms | 12.62ms (invert_displacement) | 0.09x | 35.7MB | 2.1MB | 17x |
| 64^3 | 2.82ms | 16.64ms (invert_displacement) | 0.17x | 37.7MB | 4.2MB | 9x |
| 96^3 | 11.12ms | 18.63ms (invert_displacement) | 0.60x | 105.1MB | 16.8MB | 6x |
| 128^3 | — | — | fidelity_failed | — | — | — |

- **Speed:** nitrix wins 4/4 sizes; at the largest `96^3`, nitrix 1.68x ahead.
- **Projected OOM (≈24GB):** nitrix ~202.0 Melem vs best baseline ~1266 Melem (~6x more headroom).

## jacobian_folding_penalty  (nitrix.register.jacobian_folding_penalty)  [jax-cuda12]

**Cost law.** O(N) over the voxel count N: one roll-based central-diff pass (the Jacobian) + a closed-form Sarrus determinant + relu + a mean. Stencil + reduction, memory-bandwidth-bound, GPU-pure; HBM ~ N. The relu makes the *value* data-dependent but the *cost* is not (det + relu run on every voxel). The brain-scale tier varies the volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.10ms | 0.92ms (jacobian_folding_penalty) | 0.11x | 33.9MB | 0.4MB | — |
| 48^3 | 0.11ms | 0.92ms (jacobian_folding_penalty) | 0.12x | 35.7MB | 2.1MB | 17x |
| 64^3 | 0.13ms | 0.96ms (jacobian_folding_penalty) | 0.13x | 37.7MB | 4.2MB | 9x |
| 96^3 | 0.15ms | 1.61ms (jacobian_folding_penalty) | 0.09x | 50.3MB | 16.8MB | 3x |
| 128^3 | 0.24ms | 8.90ms (jacobian_folding_penalty) | 0.03x | 67.1MB | 33.6MB | 2x |
| 160^3 | 0.27ms | 18.44ms (jacobian_folding_penalty) | 0.01x | 83.9MB | 67.1MB | 1x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 67.78x ahead.
- **Projected OOM (≈24GB):** nitrix ~1171.8 Melem vs best baseline ~1465 Melem (~1x more headroom).

## laplacian_eigenmap  (nitrix.graph.laplacian_eigenmap)  [jax-cuda12]

**Cost law.** dense O(n^3) eigh / O(n^2) backend+operator -> infeasible at n~100k (~40 GB dense); sparse lobpcg O(iters*nnz) fwd + O(nnz*k) differentiable backward -> scales (fsaverage6/7), and is the only differentiable option (scipy/cupy eigsh have no gradient).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| n=1024 dense | — | 31.34ms (eigsh) | skipped | — | 4.2MB | — |
| n=1024 dense k32 | — | 81.87ms (eigsh) | skipped | — | 4.2MB | — |
| n=2048 dense | — | 45.91ms (eigsh) | skipped | — | 16.8MB | — |
| n=2048 ell | 146.31ms | 46.20ms (eigsh) | 3.17x | 121.0MB | 20.2MB | 6x |
| n=4096 ell | 631.83ms | 81.93ms (eigsh) | 7.71x | 101.1MB | 80.3MB | 1x |
| n=10242 ell | 42.59ms | 609.07ms (eigsh) | 0.07x | 143.3MB | 8.4MB | 17x |
| n=40962 ell | 23.55ms | 1002.00ms (eigsh) | 0.02x | 153.9MB | 33.6MB | 5x |
| n=120000 ell | 47.35ms | 3273.97ms (eigsh) | 0.01x | 204.5MB | 67.1MB | 3x |

- **Speed:** nitrix wins 3/5 sizes; baseline ahead at `n=4096 ell` 7.71x, `n=2048 ell` 3.17x; at the largest `n=120000 ell`, nitrix 69.15x ahead.
- **Projected OOM (≈24GB):** nitrix ~225.4 Melem vs best baseline ~687 Melem (~3x more headroom).
- **Dispatch note (not a scale risk):** nitrix `skipped` at `n=1024 dense`, `n=1024 dense k32`, `n=2048 dense` (the default path is unavailable on this platform -- e.g. the cuSolver eigh block -- while the reference ran).

## largest_connected_component  (nitrix.morphology.largest_connected_component)  [jax-cuda12]

**Cost law.** connected_components (pointer-jumping label propagation, O(log d) passes for diameter d) + a bincount/argmax over labels. Global, GPU-pure; steady ~ N log(d). MEASURED (L4, 48^3): like connected_components, nitrix lags cupyx label here (cupyx ~2x) -- a shared kernel/algorithm candidate. The size tier varies N.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.70ms | 0.70ms (largest_cc) | 0.99x | 33.6MB | 0.0MB | — |
| 48^3 | 1.38ms | 0.85ms (largest_cc) | 1.63x | 33.7MB | 0.1MB | — |
| 64^3 | 2.24ms | 1.02ms (largest_cc) | 2.20x | 33.8MB | 0.3MB | — |
| 96^3 | 2.21ms | 1.07ms (largest_cc) | 2.07x | 34.4MB | 0.9MB | — |
| 128^3 | 7.20ms | 1.06ms (largest_cc) | 6.79x | 35.7MB | 2.1MB | 17x |
| 160^3 | 18.60ms | 1.20ms (largest_cc) | 15.56x | 79.5MB | 4.2MB | 19x |

- **Speed:** nitrix wins 1/6 sizes; baseline ahead at `160^3` 15.56x, `128^3` 6.79x, `64^3` 2.20x, `96^3` 2.07x (+1 more); at the largest `160^3`, baseline 15.56x ahead.
- **Projected OOM (≈24GB):** nitrix ~1236.6 Melem vs best baseline ~23438 Melem (~19x more headroom).

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

## max_pool_with_indices_nd  (nitrix.morphology.max_pool_with_indices_nd)  [jax-cuda12]

**Cost law.** O(N) over the input voxel count N (B*C*d^3): one windowed max + a windowed argmax per non-overlapping 2^3 block. Embarrassingly parallel, memory-bandwidth-bound, GPU-pure; the argmax ~doubles the cost over a max-only pool (~2.6x on the L4). The tier varies N.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.12ms | 0.16ms (max_pool) | 0.74x | 34.5MB | 0.8MB | — |
| 48^3 | 0.13ms | 0.23ms (max_pool) | 0.56x | 38.4MB | 4.2MB | 9x |
| 64^3 | 0.19ms | 0.37ms (max_pool) | 0.50x | 43.5MB | 8.4MB | 5x |
| 96^3 | 0.36ms | 1.16ms (max_pool) | 0.31x | 106.0MB | 33.6MB | 3x |
| 128^3 | 1.05ms | 2.63ms (max_pool) | 0.40x | 146.8MB | 67.1MB | 2x |
| 160^3 | 1.91ms | 5.19ms (max_pool) | 0.37x | 293.0MB | 134.2MB | 2x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 2.72x ahead.
- **Projected OOM (≈24GB):** nitrix ~335.5 Melem vs best baseline ~732 Melem (~2x more headroom).

## max_unpool_nd  (nitrix.morphology.max_unpool_nd)  [jax-cuda12]

**Cost law.** O(N) over the output voxel count N (B*C*d^3): allocate a zeroed grid and scatter one value per pooled position to its flat index. A scatter (data-dependent writes), memory-bandwidth-bound, GPU-pure; HBM ~ N. The size tier varies the output volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.11ms | 0.23ms (max_unpool) | 0.47x | 37.2MB | 37.2MB | 1x |
| 48^3 | 0.10ms | 0.21ms (max_unpool) | 0.49x | 47.3MB | 47.3MB | 1x |
| 64^3 | 0.11ms | 0.21ms (max_unpool) | 0.54x | 64.5MB | 64.5MB | 1x |
| 96^3 | 0.21ms | 0.24ms (max_unpool) | 0.90x | 176.7MB | 176.7MB | 1x |
| 128^3 | 0.80ms | 0.75ms (max_unpool) | 1.07x | 348.1MB | 348.1MB | 1x |
| 160^3 | 1.50ms | 1.64ms (max_unpool) | 0.92x | 684.3MB | 684.3MB | 1x |

- **Speed:** nitrix wins 5/6 sizes; baseline ahead at `128^3` 1.07x; at the largest `160^3`, nitrix 1.09x ahead.
- **Projected OOM (≈24GB):** nitrix ~143.7 Melem vs best baseline ~144 Melem (~1x more headroom).

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
| V=64 | 11.30ms | — | ok | 151.1MB | — | — |
| V=256 | 11.35ms | — | ok | 151.2MB | — | — |
| V=1024 | 11.29ms | — | ok | 135.0MB | — | — |
| V=512 | 12.65ms | — | ok | 151.1MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~174.8 Melem.

## rigid_exp  (nitrix.geometry.rigid_exp)  [jax-cuda12]

**Cost law.** O(B) over the batch B, embarrassingly parallel: Rodrigues SO(3) exp (a few 3x3 matmuls) + a direct translation per transform, tiny 4x4 matrices. Throughput-bound; launch-bound at small B (GPU favoured as B grows). HBM ~ B. The batch tier varies B (cohort / per-voxel local-affine field scale).

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| b=1024 | 0.11ms | 0.25ms (rigid_exp) | 0.42x | 0.2MB | 0.0MB | — |
| b=16384 | 0.12ms | 0.71ms (rigid_exp) | 0.17x | 34.2MB | 0.4MB | — |
| b=65536 | 0.12ms | 1.00ms (rigid_exp) | 0.12x | 36.7MB | 2.1MB | 18x |
| b=262144 | 0.22ms | 1.42ms (rigid_exp) | 0.15x | 75.5MB | 8.4MB | 9x |
| b=1048576 | 1.54ms | 8.15ms (rigid_exp) | 0.19x | 251.7MB | 33.6MB | 8x |

- **Speed:** nitrix wins 5/5 sizes; at the largest `b=1048576`, nitrix 5.31x ahead.
- **Projected OOM (≈24GB):** nitrix ~100.0 Melem vs best baseline ~750 Melem (~8x more headroom).

## rigid_log  (nitrix.geometry.rigid_log)  [jax-cuda12]

**Cost law.** O(B) over the batch B, embarrassingly parallel: principal SO(3) log (trace -> angle, off-diagonals -> axis) + the translation column per transform, tiny 4x4 matrices. Throughput-bound; HBM ~ B. The batch tier varies B.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| b=1024 | 0.10ms | 0.09ms (rigid_log) | 1.09x | 0.1MB | 0.1MB | — |
| b=16384 | 0.10ms | 0.42ms (rigid_log) | 0.25x | 2.2MB | 1.0MB | 2x |
| b=65536 | 0.11ms | 0.41ms (rigid_log) | 0.26x | 8.7MB | 4.2MB | 2x |
| b=262144 | 0.12ms | 0.42ms (rigid_log) | 0.30x | 34.6MB | 16.8MB | 2x |
| b=1048576 | 0.92ms | 2.20ms (rigid_log) | 0.42x | 138.4MB | 67.1MB | 2x |

- **Speed:** nitrix wins 4/5 sizes; baseline ahead at `b=1024` 1.09x; at the largest `b=1048576`, nitrix 2.40x ahead.
- **Projected OOM (≈24GB):** nitrix ~181.8 Melem vs best baseline ~375 Melem (~2x more headroom).

## rigid_register  (nitrix.register.rigid_register)  [jax-cuda12]

**Cost law.** post loop-roll (lax.scan): COMPILE ~flat in iterations AND volume (XLA compiles the per-iteration op graph, not an unrolled chain) -- ~4-11 s across configs/sizes (was 16-211 s unrolled). STEADY is the headline ~ iterations x P x N: each LM iter assembles the small-P normal equations J^TJ (P=6; ~P forward warp-passes + a P x P solve). GPU steady is overhead-bound (~flat) below ~48^3 then compute-bound (~N); the GPU/CPU speedup climbs from ~4x (24^3) to a brain-scale plateau ~25x. HBM: lighter than demons, but cold peak_hbm is autotune-contaminated at large N -- no OOM projection (see reports/REGISTRATION_SCALING.md). Bias: the size tier fixes (levels=2, iters=20); real pipelines raise levels with resolution.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 48x48x48 | 5.18ms | — | ok | 203.4MB | — | — |
| 96x96x96 | 26.54ms | — | ok | 1279.6MB | — | — |
| 96x96x96 world | 44.00ms | — | ok | 460.7MB | — | — |
| mni152 2mm | 22.83ms | — | ok | 1605.7MB | — | — |
| 128x128x128 | 42.74ms | — | ok | 8741.5MB | — | — |
| 128x128x128 world | 115.92ms | — | ok | 566.2MB | — | — |
| 160x160x160 | 111.01ms | — | ok | 13643.8MB | — | — |
| 192x192x192 | 205.37ms | — | ok | 1338.3MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~126.9 Melem.

## spatial_gradient  (nitrix.geometry.spatial_gradient)  [jax-cuda12]

**Cost law.** O(N) over the voxel count N: one roll-based central-diff pass per spatial axis (ndim passes), each a shift + subtract. Pure stencil, memory-bandwidth-bound and GPU-pure (no solver); HBM ~ N (the ndim-component output). The size tier varies the volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| 32^3 | 0.16ms | 0.28ms (spatial_gradient) | 0.58x | 134.6MB | 0.1MB | — |
| 48^3 | 0.16ms | 0.57ms (spatial_gradient) | 0.29x | 202.7MB | 0.4MB | — |
| 64^3 | 0.17ms | 1.14ms (spatial_gradient) | 0.15x | 615.6MB | 1.0MB | 587x |
| 96^3 | 0.24ms | 0.62ms (spatial_gradient) | 0.38x | 1305.3MB | 4.2MB | 311x |
| 128^3 | 0.47ms | 1.37ms (spatial_gradient) | 0.35x | 8749.8MB | 8.4MB | 1043x |
| 160^3 | 1.17ms | 2.86ms (spatial_gradient) | 0.41x | 13643.8MB | 16.8MB | 813x |

- **Speed:** nitrix wins 6/6 sizes; at the largest `160^3`, nitrix 2.44x ahead.
- **Projected OOM (≈24GB):** nitrix ~7.2 Melem vs best baseline ~5859 Melem (~813x more headroom).

## volreg  (nitrix.register.volreg)  [jax-cuda12]

**Cost law.** STEADY ~ T x iters x N per-frame, but the reference work (pyramid, inverse-compositional steepest-descent + Hessian) is hoisted once and the T frames are vmap-batched behind ONE compile -- so nitrix-GPU stays sublinear in T once the batch fills the device, while ANTs is T sequential CPU registrations (~T x 60 ms). The GPU:CPU gap should GROW with T (the batching/amortisation story); the honest CPU bar is the FAST community tool (3dvolreg / mcflirt), I/O-floor-subtracted, not the slower ANTs (timed out at T=500). HBM ~ T*N (realigned series + vmap working set) -- the binding constraint; OOM at the top is reported as signal. Size tier varies T (headline) + volume.

| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | base HBM | HBM x |
|---|---|---|---|---|---|---|
| T8 32x32x32 | 1.06ms | — | ok | 135.5MB | — | — |
| T16 32x32x32 | 1.45ms | — | ok | 136.3MB | — | — |
| T32 32x32x32 | 2.25ms | — | ok | 138.4MB | — | — |
| T50 48x48x48 | 74.52ms | — | ok | 334.8MB | — | — |
| T100 48x48x48 | 157.10ms | — | ok | 625.3MB | — | — |
| T200 48x48x48 | 338.05ms | — | ok | 1070.7MB | — | — |
| T100 64x64x64 | 423.46ms | — | ok | 1283.5MB | — | — |
| T100 80x80x80 | 1048.37ms | — | ok | 2415.9MB | — | — |
| T500 48x48x48 | 885.55ms | — | ok | 2483.4MB | — | — |

- **Projected OOM (≈24GB):** nitrix ~534.4 Melem.

