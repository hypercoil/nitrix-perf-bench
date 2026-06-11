# Economic verdict: nitrix-GPU vs the CPU gold standard

A GPU-hour costs **~4x** a CPU-hour on the major clouds (an L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs a comparable general-purpose vCPU instance ~$0.18/hr, 2026). So a nitrix-GPU result is *economically favorable* only when it beats the CPU gold standard by MORE than 4x -- an incremental GPU win is **not** a win once a real user pays the GPU premium. Tunable via `--cost-multiple`.

- **amortized** = CPU walltime / nitrix-GPU steady (compile amortised over many subjects / frames).
- **single-run** = CPU walltime / (nitrix-GPU steady + GPU compile) -- one run, cold.
- **verdict**: `favorable` (both >= bar) / `favorable (amortized only)` (the compile is the gate -- amortise it over the cohort) / `not multiplicative enough` (a real GPU win, but < bar -- so NOT a win by the cost test).

**Caveats (read with care):** the CPU domain tools (ANTs / dipy) run a FIXED internal schedule and ignore our `(levels, iters)`, so the verdict is meaningful across the **size / T tier**, not the dev configs; nitrix runs a fixed-iteration scan while ANTs / dipy early-exit on convergence (a wall-clock economic read, not a per-iteration claim); **time only** (HBM excluded -- cold peak is autotune-contaminated, see `SCALING.md`). For volreg the CPU bar is the **community realignment standard** -- AFNI `3dvolreg` / FSL `mcflirt` (fast, hand-optimised C), **I/O-floor-subtracted** (`compute = tool - the matching 3dcalc/fslmaths no-op`); ANTs `motion_correction` is kept only as a slow reference (timed out at T=500).

## affine_register  (nitrix.register.affine_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 26.4 ms | 10.14 s | 520.8 ms (ants.registration) | 19.8x | 0.1x | favorable (amortized only) |
| 96x96x96 world | 78.4 ms | 15.00 s | 273.7 ms (ants.registration) | 3.5x | 0.0x | not multiplicative enough |
| mni152 2mm | 36.6 ms | 10.63 s | 270.0 ms (ants.registration) | 7.4x | 0.0x | favorable (amortized only) |
| 128x128x128 | 69.7 ms | 12.19 s | 1.15 s (ants.registration) | 16.5x | 0.1x | favorable (amortized only) |
| 128x128x128 world | 198.1 ms | 20.51 s | 470.3 ms (ants.registration) | 2.4x | 0.0x | not multiplicative enough |
| 160x160x160 | 138.9 ms | 12.76 s | 1.46 s (ants.registration) | 10.5x | 0.1x | favorable (amortized only) |
| 192x192x192 | 244.6 ms | 28.86 s | 980.2 ms (ants.registration) | 4.0x | 0.0x | favorable (amortized only) |

- **5/7** size(s) favorable at 4x; best amortized **19.8x** at `96x96x96` (single-run 0.1x).

## bbr_register  (nitrix.register.bbr_register)

> No CPU **domain** tool for this op -- the CPU bar is nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium for *this* op).

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| N5000 64x64x64 | 9.6 ms | 7.64 s | 23.0 ms (nitrix-CPU) | 2.4x | 0.0x | not multiplicative enough |
| N20000 64x64x64 | 3.4 ms | 7.80 s | 43.8 ms (nitrix-CPU) | 12.7x | 0.0x | favorable (amortized only) |
| N80000 64x64x64 | 5.9 ms | 6.15 s | 174.3 ms (nitrix-CPU) | 29.7x | 0.0x | favorable (amortized only) |

- **2/3** size(s) favorable at 4x; best amortized **29.7x** at `N80000 64x64x64` (single-run 0.0x).

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

## diffeomorphic_demons  (nitrix.register.diffeomorphic_demons_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 79.5 ms | 9.91 s | 2.23 s (simpleitk.demons) | 28.1x | 0.2x | favorable (amortized only) |
| 96x96x96 aniso1x1x3 | 84.2 ms | 12.08 s | 2.25 s (simpleitk.demons) | 26.8x | 0.2x | favorable (amortized only) |
| mni152 2mm | 117.7 ms | 18.19 s | 2.72 s (simpleitk.demons) | 23.1x | 0.1x | favorable (amortized only) |
| 128x128x128 | 295.3 ms | 30.90 s | 5.64 s (simpleitk.demons) | 19.1x | 0.2x | favorable (amortized only) |
| 128x128x128 aniso1x1x3 | 301.4 ms | 52.03 s | 5.69 s (simpleitk.demons) | 18.9x | 0.1x | favorable (amortized only) |
| 160x160x160 | 647.5 ms | 30.26 s | 11.45 s (simpleitk.demons) | 17.7x | 0.4x | favorable (amortized only) |

- **6/6** size(s) favorable at 4x; best amortized **28.1x** at `96x96x96` (single-run 0.2x).

## greedy_syn_register  (nitrix.register.greedy_syn_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 64x64x64 | 178.9 ms | 21.97 s | 1.10 s (ants.registration) | 6.2x | 0.0x | favorable (amortized only) |
| 64x64x64 aniso1x1x3 | 205.6 ms | 22.74 s | 2.11 s (ants.registration) | 10.3x | 0.1x | favorable (amortized only) |
| 96x96x96 | 741.8 ms | 21.65 s | 3.40 s (ants.registration) | 4.6x | 0.2x | favorable (amortized only) |
| 96x96x96 aniso1x1x3 | 775.9 ms | 25.12 s | 6.43 s (ants.registration) | 8.3x | 0.2x | favorable (amortized only) |
| mni152 2mm | 1.06 s | 38.33 s | 4.51 s (ants.registration) | 4.2x | 0.1x | favorable (amortized only) |
| 128x128x128 | 2.46 s | 45.23 s | 6.32 s (ants.registration) | 2.6x | 0.1x | not multiplicative enough |

- **5/6** size(s) favorable at 4x; best amortized **10.3x** at `64x64x64 aniso1x1x3` (single-run 0.1x).

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

## rigid_register  (nitrix.register.rigid_register)

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| 96x96x96 | 20.2 ms | 5.61 s | 330.7 ms (ants.registration) | 16.3x | 0.1x | favorable (amortized only) |
| 96x96x96 world | 49.2 ms | 13.31 s | 260.4 ms (ants.registration) | 5.3x | 0.0x | favorable (amortized only) |
| mni152 2mm | 29.2 ms | 7.19 s | 252.8 ms (ants.registration) | 8.7x | 0.0x | favorable (amortized only) |
| 128x128x128 | 61.3 ms | 7.81 s | 381.4 ms (ants.registration) | 6.2x | 0.0x | favorable (amortized only) |
| 128x128x128 world | 137.3 ms | 17.29 s | 429.3 ms (ants.registration) | 3.1x | 0.0x | not multiplicative enough |
| 160x160x160 | 125.2 ms | 8.54 s | 731.8 ms (ants.registration) | 5.8x | 0.1x | favorable (amortized only) |
| 192x192x192 | 225.8 ms | 26.61 s | 965.4 ms (ants.registration) | 4.3x | 0.0x | favorable (amortized only) |

- **6/7** size(s) favorable at 4x; best amortized **16.3x** at `96x96x96` (single-run 0.1x).

## volreg  (nitrix.register.volreg)

> CPU times for the CLI tools (AFNI/FSL) are **I/O-subtracted**: `compute = tool wall-clock - the matching no-op` (`3dcalc`/`fslmaths` identity = the NIfTI round-trip nitrix never pays). Raw and floor shown in the tool cell.

| size | GPU steady | GPU compile | CPU compute (tool) | amortized | single-run | verdict |
|---|---|---|---|---|---|---|
| T50 48x48x48 | 81.2 ms | 6.69 s | 1.02 s (fsl.mcflirt; 3.23 s−2.22 s io) | 12.5x | 0.1x | favorable (amortized only) |
| T100 48x48x48 | 172.8 ms | 10.00 s | 2.01 s (fsl.mcflirt; 6.21 s−4.20 s io) | 11.6x | 0.2x | favorable (amortized only) |
| T200 48x48x48 | 355.6 ms | 10.13 s | 3.92 s (fsl.mcflirt; 12.19 s−8.27 s io) | 11.0x | 0.4x | favorable (amortized only) |
| T100 64x64x64 | 443.5 ms | 10.04 s | 4.63 s (fsl.mcflirt; 14.23 s−9.60 s io) | 10.4x | 0.4x | favorable (amortized only) |
| T100 80x80x80 | 1.07 s | 12.09 s | 8.42 s (fsl.mcflirt; 26.54 s−18.12 s io) | 7.8x | 0.6x | favorable (amortized only) |
| T500 48x48x48 | 904.8 ms | 12.49 s | 10.17 s (fsl.mcflirt; 29.77 s−19.60 s io) | 11.2x | 0.8x | favorable (amortized only) |

- **6/6** size(s) favorable at 4x; best amortized **12.5x** at `T50 48x48x48` (single-run 0.1x).

