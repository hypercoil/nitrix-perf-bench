# Economic verdict: nitrix-GPU vs the CPU gold standard

A GPU-hour costs **~4x** a CPU-hour on the major clouds (an L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs a comparable general-purpose vCPU instance ~$0.18/hr, 2026). So a nitrix-GPU result is *economically favorable* only when it beats the CPU gold standard by MORE than 4x -- an incremental GPU win is **not** a win once a real user pays the GPU premium. Tunable via `--cost-multiple`.

- **amortized** = CPU walltime / nitrix-GPU steady (compile amortised over many subjects / frames).
- **single-run** = CPU walltime / (nitrix-GPU steady + GPU compile) -- one run, cold.
- **verdict**: `favorable` (both >= bar) / `favorable (amortized only)` (the compile is the gate -- amortise it over the cohort) / `not multiplicative enough` (a real GPU win, but < bar -- so NOT a win by the cost test).

**Caveats (read with care):** the CPU domain tools (ANTs / dipy) run a FIXED internal schedule and ignore our `(levels, iters)`, so the verdict is meaningful across the **size / T tier**, not the dev configs; nitrix runs a fixed-iteration scan while ANTs / dipy early-exit on convergence (a wall-clock economic read, not a per-iteration claim); **time only** (HBM excluded -- cold peak is autotune-contaminated, see `SCALING.md`). volreg's ANTs ref is **provisional** (not the community realignment standard -- AFNI `3dvolreg` / FSL `mcflirt`, fast, are planned).

## affine_register  (nitrix.register.affine_register)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 125.0 ms | 12.12 s | 1.60 s (ants.registration) | 12.8x | 0.1x | favorable (amortized only) |
| 96x96x96 world | 125.4 ms | 16.02 s | 625.9 ms (ants.registration) | 5.0x | 0.0x | favorable (amortized only) |
| 128x128x128 | 302.8 ms | 8.91 s | 727.0 ms (ants.registration) | 2.4x | 0.1x | not multiplicative enough |
| 128x128x128 world | 312.5 ms | 31.29 s | 863.1 ms (ants.registration) | 2.8x | 0.0x | not multiplicative enough |
| 160x160x160 | 556.2 ms | 22.61 s | 1.00 s (ants.registration) | 1.8x | 0.0x | not multiplicative enough |
| 192x192x192 | 1.04 s | 24.72 s | 1.39 s (ants.registration) | 1.3x | 0.1x | not multiplicative enough |

- **2/6** size(s) favorable at 4x; best amortized **12.8x** at `96x96x96` (single-run 0.1x).

## bbr_register  (nitrix.register.bbr_register)

> No CPU **domain** tool for this op -- the CPU bar is nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium for *this* op).

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| N5000 64x64x64 | 1.4 ms | 7.96 s | 22.6 ms (nitrix-CPU) | 16.7x | 0.0x | favorable (amortized only) |
| N20000 64x64x64 | 6.9 ms | 9.21 s | 45.7 ms (nitrix-CPU) | 6.7x | 0.0x | favorable (amortized only) |
| N80000 64x64x64 | 17.2 ms | 6.38 s | 172.0 ms (nitrix-CPU) | 10.0x | 0.0x | favorable (amortized only) |

- **3/3** size(s) favorable at 4x; best amortized **16.7x** at `N5000 64x64x64` (single-run 0.0x).

## conditionalcorr  (nitrix.stats.conditionalcorr)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d16 obs4096 | 0.9 ms | 1.37 s | 65.6 ms (numpy.conditionalcorr) | 71.7x | 0.0x | favorable (amortized only) |
| c2048 d32 obs8192 | 6.7 ms | 1.42 s | 563.5 ms (numpy.conditionalcorr) | 84.1x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **84.1x** at `c2048 d32 obs8192` (single-run 0.4x).

## conditionalcov  (nitrix.stats.conditionalcov)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d16 obs4096 | 0.9 ms | 1.33 s | 81.0 ms (numpy.conditionalcov) | 89.1x | 0.1x | favorable (amortized only) |
| c2048 d32 obs8192 | 7.5 ms | 1.46 s | 529.1 ms (numpy.conditionalcov) | 70.9x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **89.1x** at `c1024 d16 obs4096` (single-run 0.1x).

## diffeomorphic_demons  (nitrix.register.diffeomorphic_demons_register)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 80.4 ms | 17.34 s | 2.34 s (simpleitk.demons) | 29.1x | 0.1x | favorable (amortized only) |
| 96x96x96 aniso1x1x3 | 83.8 ms | 26.49 s | 2.22 s (simpleitk.demons) | 26.5x | 0.1x | favorable (amortized only) |
| 128x128x128 | 295.8 ms | 36.84 s | 6.37 s (simpleitk.demons) | 21.5x | 0.2x | favorable (amortized only) |
| 128x128x128 aniso1x1x3 | 302.7 ms | 52.80 s | 5.87 s (simpleitk.demons) | 19.4x | 0.1x | favorable (amortized only) |
| 160x160x160 | 650.2 ms | 36.79 s | 11.23 s (simpleitk.demons) | 17.3x | 0.3x | favorable (amortized only) |

- **5/5** size(s) favorable at 4x; best amortized **29.1x** at `96x96x96` (single-run 0.1x).

## greedy_syn_register  (nitrix.register.greedy_syn_register)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 64x64x64 | 171.4 ms | 13.99 s | 887.2 ms (ants.registration) | 5.2x | 0.1x | favorable (amortized only) |
| 64x64x64 aniso1x1x3 | 195.7 ms | 16.34 s | 1.89 s (ants.registration) | 9.7x | 0.1x | favorable (amortized only) |
| 96x96x96 | 742.6 ms | 17.74 s | 2.78 s (ants.registration) | 3.7x | 0.2x | not multiplicative enough |
| 96x96x96 aniso1x1x3 | 768.6 ms | 21.02 s | 5.94 s (ants.registration) | 7.7x | 0.3x | favorable (amortized only) |
| 128x128x128 | 2.46 s | 39.36 s | 5.66 s (ants.registration) | 2.3x | 0.1x | not multiplicative enough |

- **3/5** size(s) favorable at 4x; best amortized **9.7x** at `64x64x64 aniso1x1x3` (single-run 0.1x).

## pairedcorr  (nitrix.stats.pairedcorr)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d1024 obs4096 | 2.4 ms | 712.6 ms | 71.4 ms (numpy.pairedcorr) | 30.1x | 0.1x | favorable (amortized only) |
| c2048 d2048 obs8192 | 19.2 ms | 659.5 ms | 523.1 ms (numpy.pairedcorr) | 27.3x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **30.1x** at `c1024 d1024 obs4096` (single-run 0.1x).

## pairedcov  (nitrix.stats.pairedcov)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| c1024 d1024 obs4096 | 0.9 ms | 526.6 ms | 53.8 ms (numpy.pairedcov) | 57.6x | 0.1x | favorable (amortized only) |
| c2048 d2048 obs8192 | 6.0 ms | 606.7 ms | 484.7 ms (numpy.pairedcov) | 81.1x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **81.1x** at `c2048 d2048 obs8192` (single-run 0.8x).

## pca_fit  (nitrix.stats.pca_fit)

> No CPU **domain** tool for this op -- the CPU bar is nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium for *this* op).

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 4096x1024 k32 | 12.1 ms | 980.0 ms | 937.7 ms (sklearn.PCA) | 77.4x | 0.9x | favorable (amortized only) |
| 8192x2048 k32 | 44.9 ms | 1.66 s | 1.36 s (nitrix-CPU) | 30.3x | 0.8x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **77.4x** at `4096x1024 k32` (single-run 0.9x).

## pca_inverse_transform  (nitrix.stats.pca_inverse_transform)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 65536x1024 k64 | 1.3 ms | 249.7 ms | 339.9 ms (numpy.matmul) | 261.5x | 1.4x | favorable (amortized only) |
| 131072x512 k64 | 1.4 ms | 306.9 ms | 335.1 ms (numpy.matmul) | 238.3x | 1.1x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **261.5x** at `65536x1024 k64` (single-run 1.4x).

## pca_transform  (nitrix.stats.pca_transform)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 65536x1024 k64 | 3.7 ms | 737.9 ms | 243.4 ms (numpy.matmul) | 65.7x | 0.3x | favorable (amortized only) |
| 131072x512 k64 | 3.7 ms | 630.5 ms | 255.8 ms (numpy.matmul) | 68.5x | 0.4x | favorable (amortized only) |

- **2/2** size(s) favorable at 4x; best amortized **68.5x** at `131072x512 k64` (single-run 0.4x).

## rigid_register  (nitrix.register.rigid_register)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 91.8 ms | 9.17 s | 344.6 ms (ants.registration) | 3.8x | 0.0x | not multiplicative enough |
| 96x96x96 world | 101.3 ms | 12.21 s | 305.1 ms (ants.registration) | 3.0x | 0.0x | not multiplicative enough |
| 128x128x128 | 227.9 ms | 12.22 s | 1.35 s (ants.registration) | 5.9x | 0.1x | favorable (amortized only) |
| 128x128x128 world | 231.7 ms | 20.12 s | 500.9 ms (ants.registration) | 2.2x | 0.0x | not multiplicative enough |
| 160x160x160 | 415.4 ms | 11.28 s | 777.0 ms (ants.registration) | 1.9x | 0.1x | not multiplicative enough |
| 192x192x192 | 808.5 ms | 21.26 s | 1.02 s (ants.registration) | 1.3x | 0.0x | not multiplicative enough |

- **1/6** size(s) favorable at 4x; best amortized **5.9x** at `128x128x128` (single-run 0.1x).

## volreg  (nitrix.register.volreg)

| size | GPU steady | GPU compile | CPU gold (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| T50 48x48x48 | 81.2 ms | 8.24 s | 4.39 s (ants.motion_correction) | 54.1x | 0.5x | favorable (amortized only) |
| T100 48x48x48 | 172.7 ms | 8.05 s | 10.09 s (ants.motion_correction) | 58.4x | 1.2x | favorable (amortized only) |
| T200 48x48x48 | 355.5 ms | 8.14 s | 23.59 s (ants.motion_correction) | 66.3x | 2.8x | favorable (amortized only) |
| T100 64x64x64 | 443.4 ms | 9.38 s | 24.10 s (ants.motion_correction) | 54.4x | 2.5x | favorable (amortized only) |
| T100 80x80x80 | 1.09 s | 10.89 s | 29.68 s (ants.motion_correction) | 27.3x | 2.5x | favorable (amortized only) |
| T500 48x48x48 | 904.5 ms | 9.89 s | 85.10 s (ants.motion_correction) | 94.1x | 7.9x | favorable |

- **6/6** size(s) favorable at 4x; best amortized **94.1x** at `T500 48x48x48` (single-run 7.9x).

