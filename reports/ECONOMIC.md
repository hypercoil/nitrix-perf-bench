# Economic verdict: nitrix-GPU vs the CPU gold standard

A GPU-hour costs **~4x** a CPU-hour on the major clouds (an L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs a comparable general-purpose vCPU instance ~$0.18/hr, 2026). So a nitrix-GPU result is *economically favorable* only when it beats the CPU gold standard by MORE than 4x -- an incremental GPU win is **not** a win once a real user pays the GPU premium. Tunable via `--cost-multiple`.

- **amortized** = CPU walltime / nitrix-GPU steady (compile amortised over many subjects / frames).
- **single-run** = CPU walltime / (nitrix-GPU steady + GPU compile) -- one run, cold.
- **verdict**: `favorable` (both >= bar) / `favorable (amortized only)` (the compile is the gate -- amortise it over the cohort) / `not multiplicative enough` (a real GPU win, but < bar -- so NOT a win by the cost test).

**Caveats (read with care):** the CPU domain tools (ANTs / dipy) run a FIXED internal schedule and ignore our `(levels, iters)`, so the verdict is meaningful across the **size / T tier**, not the dev configs; nitrix runs a fixed-iteration scan while ANTs / dipy early-exit on convergence (a wall-clock economic read, not a per-iteration claim); **time only** (HBM excluded -- cold peak is autotune-contaminated, see `SCALING.md`). For volreg the CPU bar is the **community realignment standard** -- AFNI `3dvolreg` / FSL `mcflirt` (fast, hand-optimised C), **I/O-floor-subtracted** (`compute = tool - the matching 3dcalc/fslmaths no-op`); ANTs `motion_correction` is kept only as a slow reference (timed out at T=500).

## affine_exp  (nitrix.geometry.affine_exp)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| b=262144 | 1.4 ms | 422.5 ms | 495.1 ms (numpy.affine_exp) | 345.1x | 1.2x | favorable (amortized only) |
| b=1048576 | 13.5 ms | 343.5 ms | 3.18 s (numpy.affine_exp) | 235.3x | 8.9x | favorable |

- **2/2** size(s) favorable at 4x; best amortized **345.1x** at `b=262144` (single-run 1.2x).

## affine_register  (nitrix.register.affine_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 35.6 ms | 34.34 s | 4.19 s (ants.registration) | 117.5x | 0.1x | favorable (amortized only) |
| 96x96x96 world | 73.2 ms | 31.12 s | 4.40 s (ants.registration) | 60.0x | 0.1x | favorable (amortized only) |
| mni152 2mm | 27.5 ms | 26.03 s | 818.0 ms (ants.registration) | 29.8x | 0.0x | favorable (amortized only) |
| 128x128x128 | 52.2 ms | 35.82 s | 4.44 s (ants.registration) | 85.0x | 0.1x | favorable (amortized only) |
| 128x128x128 world | 176.4 ms | 44.12 s | 1.14 s (ants.registration) | 6.5x | 0.0x | favorable (amortized only) |
| 160x160x160 | 119.5 ms | 36.40 s | 4.78 s (ants.registration) | 40.0x | 0.1x | favorable (amortized only) |
| 192x192x192 | 224.0 ms | 30.93 s | 1.58 s (ants.registration) | 7.1x | 0.1x | favorable (amortized only) |

- **7/7** size(s) favorable at 4x; best amortized **117.5x** at `96x96x96` (single-run 0.1x).

## bbr_register  (nitrix.register.bbr_register)

> No CPU **domain** tool for this op -- the CPU bar is nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium for *this* op).

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| N5000 64x64x64 | 1.4 ms | 6.44 s | 23.4 ms (nitrix-CPU) | 16.6x | 0.0x | favorable (amortized only) |
| N20000 64x64x64 | 5.3 ms | 6.75 s | 45.6 ms (nitrix-CPU) | 8.6x | 0.0x | favorable (amortized only) |
| N80000 64x64x64 | 10.1 ms | 4.46 s | 169.7 ms (nitrix-CPU) | 16.7x | 0.0x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **16.7x** at `N80000 64x64x64` (single-run 0.0x).

## bending_energy  (nitrix.register.bending_energy)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 2.6 ms | 2.84 s | 395.6 ms (numpy.bending_energy) | 152.5x | 0.1x | favorable (amortized only) |
| 128^3 | 7.3 ms | 3.84 s | 1.03 s (numpy.bending_energy) | 141.6x | 0.3x | favorable (amortized only) |
| 160^3 | 15.2 ms | 3.68 s | 2.32 s (numpy.bending_energy) | 152.3x | 0.6x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **152.5x** at `96^3` (single-run 0.1x).

## compose_velocity  (nitrix.geometry.compose_velocity)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.3 ms | 847.3 ms | 155.5 ms (numpy.compose_velocity) | 603.9x | 0.2x | favorable (amortized only) |
| 128^3 | 1.1 ms | 921.3 ms | 526.1 ms (numpy.compose_velocity) | 498.5x | 0.6x | favorable (amortized only) |
| 160^3 | 2.0 ms | 878.4 ms | 1.56 s (numpy.compose_velocity) | 785.6x | 1.8x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **785.6x** at `160^3` (single-run 1.8x).

## conditionalcorr  (nitrix.stats.conditionalcorr)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d16 obs4096 | 0.9 ms | 1.37 s | 65.6 ms (numpy.conditionalcorr) | 71.7x | 0.0x | favorable (amortized only) |
| c2048 d32 obs8192 | 6.7 ms | 1.42 s | 563.5 ms (numpy.conditionalcorr) | 84.1x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **84.1x** at `c2048 d32 obs8192` (single-run 0.4x).

## conditionalcov  (nitrix.stats.conditionalcov)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d16 obs4096 | 0.9 ms | 1.33 s | 81.0 ms (numpy.conditionalcov) | 89.1x | 0.1x | favorable (amortized only) |
| c2048 d32 obs8192 | 7.5 ms | 1.46 s | 529.1 ms (numpy.conditionalcov) | 70.9x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **89.1x** at `c1024 d16 obs4096` (single-run 0.1x).

## connected_components  (nitrix.morphology.connected_components)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 1.7 ms | 697.4 ms | 11.5 ms (scipy.label) | 6.5x | 0.0x | favorable (amortized only) |
| 128^3 | 6.1 ms | 841.0 ms | 21.9 ms (scipy.label) | 3.6x | 0.0x | not multiplicative enough |
| 160^3 | 15.8 ms | 646.0 ms | 34.8 ms (scipy.label) | 2.2x | 0.1x | not multiplicative enough |

- **1/3** size(s) favorable at 4x; best amortized **6.5x** at `96^3` (single-run 0.0x).

## diffeomorphic_demons  (nitrix.register.diffeomorphic_demons_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 66.0 ms | 24.45 s | 3.58 s (simpleitk.demons) | 54.2x | 0.1x | favorable (amortized only) |
| 96x96x96 aniso1x1x3 | 51.3 ms | 23.78 s | 2.43 s (simpleitk.demons) | 47.3x | 0.1x | favorable (amortized only) |
| mni152 2mm | 69.3 ms | 31.68 s | 2.80 s (simpleitk.demons) | 40.4x | 0.1x | favorable (amortized only) |
| 128x128x128 | 281.3 ms | 27.97 s | 6.11 s (simpleitk.demons) | 21.7x | 0.2x | favorable (amortized only) |
| 128x128x128 aniso1x1x3 | 183.5 ms | 28.33 s | 6.17 s (simpleitk.demons) | 33.6x | 0.2x | favorable (amortized only) |
| 160x160x160 | 741.1 ms | 24.24 s | 11.90 s (simpleitk.demons) | 16.1x | 0.5x | favorable (amortized only) |

- **6/6** size(s) favorable at 4x; best amortized **54.2x** at `96x96x96` (single-run 0.1x).

## distance_transform_edt  (nitrix.morphology.distance_transform_edt)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.4 ms | 824.4 ms | 111.2 ms (scipy.distance_transform_edt) | 299.6x | 0.1x | favorable (amortized only) |
| 128^3 | 0.5 ms | 1.04 s | 358.7 ms (scipy.distance_transform_edt) | 752.6x | 0.3x | favorable (amortized only) |
| 160^3 | 2.0 ms | 880.6 ms | 572.0 ms (scipy.distance_transform_edt) | 289.9x | 0.6x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **752.6x** at `128^3` (single-run 0.3x).

## gradient_smoothness  (nitrix.register.gradient_smoothness)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.2 ms | 540.3 ms | 109.5 ms (numpy.gradient_smoothness) | 670.2x | 0.2x | favorable (amortized only) |
| 128^3 | 0.2 ms | 470.7 ms | 325.6 ms (numpy.gradient_smoothness) | 1438.9x | 0.7x | favorable (amortized only) |
| 160^3 | 0.4 ms | 667.1 ms | 811.5 ms (numpy.gradient_smoothness) | 2315.0x | 1.2x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **2315.0x** at `160^3` (single-run 1.2x).

## greedy_syn_register  (nitrix.register.greedy_syn_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 64x64x64 | 115.5 ms | 29.67 s | 1.73 s (ants.registration) | 15.0x | 0.1x | favorable (amortized only) |
| 64x64x64 aniso1x1x3 | 120.2 ms | 32.42 s | 2.80 s (ants.registration) | 23.3x | 0.1x | favorable (amortized only) |
| 96x96x96 | 398.7 ms | 33.55 s | 3.67 s (ants.registration) | 9.2x | 0.1x | favorable (amortized only) |
| 96x96x96 aniso1x1x3 | 419.0 ms | 32.40 s | 6.63 s (ants.registration) | 15.8x | 0.2x | favorable (amortized only) |
| mni152 2mm | 534.6 ms | 44.87 s | 6.34 s (ants.registration) | 11.9x | 0.1x | favorable (amortized only) |
| 128x128x128 | 1.24 s | 40.63 s | 6.71 s (ants.registration) | 5.4x | 0.2x | favorable (amortized only) |

- **6/6** size(s) favorable at 4x; best amortized **23.3x** at `64x64x64 aniso1x1x3` (single-run 0.1x).

## invert_displacement  (nitrix.geometry.invert_displacement)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 11.1 ms | 815.5 ms | 4.06 s (numpy.invert_displacement) | 365.4x | 4.9x | favorable |

- **1/1** size(s) favorable at 4x; best amortized **365.4x** at `96^3` (single-run 4.9x).

## jacobian_folding_penalty  (nitrix.register.jacobian_folding_penalty)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.1 ms | 1.01 s | 95.5 ms (numpy.jacobian_folding_penalty) | 649.9x | 0.1x | favorable (amortized only) |
| 128^3 | 0.2 ms | 697.5 ms | 211.4 ms (numpy.jacobian_folding_penalty) | 872.9x | 0.3x | favorable (amortized only) |
| 160^3 | 0.3 ms | 682.8 ms | 544.3 ms (numpy.jacobian_folding_penalty) | 2000.3x | 0.8x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **2000.3x** at `160^3` (single-run 0.8x).

## largest_connected_component  (nitrix.morphology.largest_connected_component)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 2.2 ms | 1.35 s | 10.4 ms (scipy.largest_cc) | 4.7x | 0.0x | favorable (amortized only) |
| 128^3 | 7.2 ms | 1.66 s | 33.3 ms (scipy.largest_cc) | 4.6x | 0.0x | favorable (amortized only) |
| 160^3 | 18.6 ms | 1.71 s | 63.7 ms (scipy.largest_cc) | 3.4x | 0.0x | not multiplicative enough |

- **2/3** size(s) favorable at 4x; best amortized **4.7x** at `96^3` (single-run 0.0x).

## max_pool_with_indices_nd  (nitrix.morphology.max_pool_with_indices_nd)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.4 ms | 412.0 ms | 135.9 ms (numpy.max_pool) | 379.4x | 0.3x | favorable (amortized only) |
| 128^3 | 1.0 ms | 456.5 ms | 352.1 ms (numpy.max_pool) | 336.2x | 0.8x | favorable (amortized only) |
| 160^3 | 1.9 ms | 410.0 ms | 700.0 ms (numpy.max_pool) | 366.1x | 1.7x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **379.4x** at `96^3` (single-run 0.3x).

## max_unpool_nd  (nitrix.morphology.max_unpool_nd)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.2 ms | 179.9 ms | 6.5 ms (numpy.max_unpool) | 30.5x | 0.0x | favorable (amortized only) |
| 128^3 | 0.8 ms | 224.1 ms | 37.5 ms (numpy.max_unpool) | 47.1x | 0.2x | favorable (amortized only) |
| 160^3 | 1.5 ms | 178.7 ms | 70.5 ms (numpy.max_unpool) | 47.0x | 0.4x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **47.1x** at `128^3` (single-run 0.2x).

## pairedcorr  (nitrix.stats.pairedcorr)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d1024 obs4096 | 2.4 ms | 712.6 ms | 71.4 ms (numpy.pairedcorr) | 30.1x | 0.1x | favorable (amortized only) |
| c2048 d2048 obs8192 | 19.2 ms | 659.5 ms | 523.1 ms (numpy.pairedcorr) | 27.3x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **30.1x** at `c1024 d1024 obs4096` (single-run 0.1x).

## pairedcov  (nitrix.stats.pairedcov)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d1024 obs4096 | 0.9 ms | 526.6 ms | 53.8 ms (numpy.pairedcov) | 57.6x | 0.1x | favorable (amortized only) |
| c2048 d2048 obs8192 | 6.0 ms | 606.7 ms | 484.7 ms (numpy.pairedcov) | 81.1x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **81.1x** at `c2048 d2048 obs8192` (single-run 0.8x).

## pca_fit  (nitrix.stats.pca_fit)

> No CPU **domain** tool for this op -- the CPU bar is nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium for *this* op).

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 4096x1024 k32 | 12.1 ms | 980.0 ms | 937.7 ms (sklearn.PCA) | 77.4x | 0.9x | favorable (amortized only) |
| 8192x2048 k32 | 44.9 ms | 1.66 s | 1.36 s (nitrix-CPU) | 30.3x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **77.4x** at `4096x1024 k32` (single-run 0.9x).

## pca_inverse_transform  (nitrix.stats.pca_inverse_transform)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 65536x1024 k64 | 1.3 ms | 249.7 ms | 339.9 ms (numpy.matmul) | 261.5x | 1.4x | favorable (amortized only) |
| 131072x512 k64 | 1.4 ms | 306.9 ms | 335.1 ms (numpy.matmul) | 238.3x | 1.1x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **261.5x** at `65536x1024 k64` (single-run 1.4x).

## pca_transform  (nitrix.stats.pca_transform)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 65536x1024 k64 | 3.7 ms | 737.9 ms | 243.4 ms (numpy.matmul) | 65.7x | 0.3x | favorable (amortized only) |
| 131072x512 k64 | 3.7 ms | 630.5 ms | 255.8 ms (numpy.matmul) | 68.5x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **68.5x** at `131072x512 k64` (single-run 0.4x).

## rigid_exp  (nitrix.geometry.rigid_exp)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| b=262144 | 0.2 ms | 577.5 ms | 60.4 ms (numpy.rigid_exp) | 274.6x | 0.1x | favorable (amortized only) |
| b=1048576 | 1.5 ms | 719.7 ms | 370.7 ms (numpy.rigid_exp) | 241.4x | 0.5x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **274.6x** at `b=262144` (single-run 0.1x).

## rigid_log  (nitrix.geometry.rigid_log)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| b=262144 | 0.1 ms | 222.8 ms | 15.8 ms (numpy.rigid_log) | 127.7x | 0.1x | favorable (amortized only) |
| b=1048576 | 0.9 ms | 217.9 ms | 74.5 ms (numpy.rigid_log) | 81.2x | 0.3x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **127.7x** at `b=262144` (single-run 0.1x).

## rigid_register  (nitrix.register.rigid_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 26.5 ms | 32.38 s | 3.93 s (ants.registration) | 148.0x | 0.1x | favorable (amortized only) |
| 96x96x96 world | 44.0 ms | 11.78 s | 777.0 ms (ants.registration) | 17.7x | 0.1x | favorable (amortized only) |
| mni152 2mm | 22.8 ms | 22.63 s | 820.1 ms (ants.registration) | 35.9x | 0.0x | favorable (amortized only) |
| 128x128x128 | 42.7 ms | 29.39 s | 4.02 s (ants.registration) | 94.0x | 0.1x | favorable (amortized only) |
| 128x128x128 world | 115.9 ms | 41.05 s | 4.18 s (ants.registration) | 36.0x | 0.1x | favorable (amortized only) |
| 160x160x160 | 111.0 ms | 24.26 s | 1.22 s (ants.registration) | 10.9x | 0.0x | favorable (amortized only) |
| 192x192x192 | 205.4 ms | 26.97 s | 1.52 s (ants.registration) | 7.4x | 0.1x | favorable (amortized only) |

- **7/7** size(s) favorable at 4x; best amortized **148.0x** at `96x96x96` (single-run 0.1x).

## spatial_gradient  (nitrix.geometry.spatial_gradient)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96^3 | 0.2 ms | 3.06 s | 8.2 ms (numpy.spatial_gradient) | 34.5x | 0.0x | favorable (amortized only) |
| 128^3 | 0.5 ms | 3.89 s | 25.9 ms (numpy.spatial_gradient) | 54.6x | 0.0x | favorable (amortized only) |
| 160^3 | 1.2 ms | 4.01 s | 53.5 ms (numpy.spatial_gradient) | 45.7x | 0.0x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **54.6x** at `128^3` (single-run 0.0x).

## volreg  (nitrix.register.volreg)

> CPU times for the CLI tools (AFNI/FSL) are **I/O-subtracted**: `compute = tool wall-clock - the matching no-op` (`3dcalc`/`fslmaths` identity = the NIfTI round-trip nitrix never pays). Raw and floor shown in the tool cell.

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| T50 48x48x48 | 74.5 ms | 11.01 s | 1.03 s (fsl.mcflirt; 3.20 s−2.17 s io) | 13.9x | 0.1x | favorable (amortized only) |
| T100 48x48x48 | 157.1 ms | 16.59 s | 3.87 s (fsl.mcflirt; 8.07 s−4.20 s io) | 24.6x | 0.2x | favorable (amortized only) |
| T200 48x48x48 | 338.1 ms | 18.07 s | 3.91 s (fsl.mcflirt; 11.90 s−7.99 s io) | 11.6x | 0.2x | favorable (amortized only) |
| T100 64x64x64 | 423.5 ms | 17.61 s | 4.56 s (fsl.mcflirt; 14.28 s−9.72 s io) | 10.8x | 0.3x | favorable (amortized only) |
| T100 80x80x80 | 1.05 s | 18.29 s | 8.47 s (fsl.mcflirt; 26.89 s−18.43 s io) | 8.1x | 0.4x | favorable (amortized only) |
| T500 48x48x48 | 885.6 ms | 15.91 s | 9.92 s (fsl.mcflirt; 29.13 s−19.21 s io) | 11.2x | 0.6x | favorable (amortized only) |

- **6/6** size(s) favorable at 4x; best amortized **24.6x** at `T100 48x48x48` (single-run 0.2x).

