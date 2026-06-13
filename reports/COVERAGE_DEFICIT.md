# nitrix-perf-bench — coverage & deficit report

> Generated from the L4 store joined with the nitrix op catalogue (`op_matrix.json`). No values are hand-edited; every ratio is read from the stored rows (SCHEMA §G).

## Coverage (runtime ops)

- **runtime ops catalogued**: 209 (+ 16 host-side constructors, apart)
- **measured** (≥1 platform): 85 / 209
- **multiplatform** (CPU + GPU): 78 / 209
- **with a strong on-target GPU ref**: 73 / 209
- **with a community-gold ref** (ANTs/FSL/…): 17 / 209
- **scaled** (ran at the declared brain-scale tier): 18 / 209 — **1** fragile (oom/timeout)
- **economically favorable** (GPU beats CPU gold by ≥ the bar): 65 / 209 — **10** not multiplicative
- **on real data** (planted or full): 3 / 209
- **marquee** ops (held to the real-data + community-baseline bar): 8 — **5** not yet meeting it
- **lagging on the GPU**: 12
- **GPU blocked upstream** (jaxlib cuSOLVER): 2
- ⚠️ **3 benchmarked case(s) absent from the catalogue** (`op_matrix.json` is stale -- invisible to the join until regenerated in nitrix): `bbr_register`, `greedy_syn_register`, `volreg`. Includes **MARQUEE** ops: `bbr_register`, `greedy_syn_register`, `volreg`.

## Lagging on the deployment target (GPU) — ranked

nitrix is slower than its strong on-target reference here (`ratio = ref/nitrix < 1`); worst first. The Pallas-kernel / algorithm candidates.

| # | op | strong GPU ref | ratio (ref/nitrix) | nitrix | note |
|---|---|---|---:|---|---|
| 1 | `nitrix.geometry.sphere_grid_unpad_2d` | cupy.sphere_grid_unpad_2d | 0.0379 | ~26.4x slower |  |
| 2 | `nitrix.graph.degree_vector` | cupy.degree | 0.177 | ~5.6x slower |  |
| 3 | `nitrix.morphology.median_filter` | cupyx.scipy.ndimage.median_filter | 0.201 | ~5.0x slower |  |
| 4 | `nitrix.linalg.linear_kernel` | cupy.linear_kernel | 0.518 | ~1.9x slower |  |
| 5 | `nitrix.geometry.spatial_transform` | cupyx.scipy.ndimage.map_coordinates | 0.557 | ~1.8x slower |  |
| 6 | `nitrix.numerics.intensity_normalize` | cupy.intensity_normalize | 0.637 | ~1.6x slower |  |
| 7 | `nitrix.metrics.ssd` | cupy.ssd | 0.673 | ~1.5x slower |  |
| 8 | `nitrix.geometry.center_of_mass_points` | cupy.center_of_mass_points | 0.698 | ~1.4x slower |  |
| 9 | `nitrix.graph.laplacian` | cupy.laplacian | 0.743 | ~1.3x slower |  |
| 10 | `nitrix.stats.pca_transform` | cupy.matmul | 0.778 | ~1.3x slower |  |
| 11 | `nitrix.stats.pca_fit` | cupy.eigh_cov | 0.962 | ~1.0x slower |  |
| 12 | `nitrix.morphology.distance_transform` | cupyx.scipy.ndimage.distance_transform_edt | 0.963 | ~1.0x slower | provisional (fast run) |

## GPU blocked — nitrix path skipped, a GPU ref works

nitrix's GPU attempt was skipped for the recorded reason below, while a strong external GPU ref **did** run ok on the GPU -- so the GPU is capable of the op but nitrix's path is not using it. For the eigh family the cause is the jaxlib cuSOLVER bug (jax-ml/jax #29042; CuPy works on identical wheels); these are benchmarked on CPU and the fix is upstream.

| op | nitrix on GPU (skipped) |
|---|---|
| `nitrix.graph.laplacian_eigenmap` | gpu_solver_unavailable |
| `nitrix.graph.diffusion_embedding` | gpu_solver_unavailable |

## Under-covered — ranked by priority

Priority is a coarse heuristic (no consumer-traffic weighting yet): **high** = unmeasured or missing a platform; **medium** = measured on both but no strong on-target GPU ref (no apples-to-apples bar).

| priority | op | coverage | ref strength | precision |
|---|---|---|---|---|
| high | `nitrix.augment.gamma_contrast` | unmeasured | none | unmeasured |
| high | `nitrix.augment.gaussian_noise` | unmeasured | none | unmeasured |
| high | `nitrix.augment.gibbs_ringing` | unmeasured | none | unmeasured |
| high | `nitrix.augment.gmm_label_to_image` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_affine_matrix` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_crop` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_flip` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_histogram_shift` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_resized_crop` | unmeasured | none | unmeasured |
| high | `nitrix.augment.random_svf_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.augment.rician_noise` | unmeasured | none | unmeasured |
| high | `nitrix.augment.simulate_bias_field` | unmeasured | none | unmeasured |
| high | `nitrix.bias.bias_field_correction` | unmeasured | none | unmeasured |
| high | `nitrix.bias.bspline_approximate` | unmeasured | none | unmeasured |
| high | `nitrix.bias.sharpen_histogram` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.affine_exp` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.affine_grid` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.affine_matrix_to_params` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.angles_to_rotation_matrix` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.apply_affine` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.compose_affine` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.compose_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.compose_velocity` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.downsample` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.fit_affine` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.gaussian_pyramid` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.invert_affine` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.invert_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.make_square_affine` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.params_to_affine_matrix` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[cubic]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[lanczos]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[multilabel]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[nearest]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.rigid_exp` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.rigid_log` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.rotation_matrix_to_angles` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.sample_at_points` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.spatial_gradient` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.spatial_transform[lanczos]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.upsample` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cg` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cho_solve` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cone_project_spd` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.delete_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.fill_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.gauss_newton` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.implicit_least_squares` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.levenberg_marquardt` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.matrix_exp` | gpu_only | floor_only | f32_only |
| high | `nitrix.linalg.mean_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_log_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.parameterised_norm` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.recondition_eigenspaces` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.solve` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.squareform` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.sym2vec` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmap` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmetric` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz_2d` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.vec2sym` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.bce_with_logits` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.correlation_ratio` | gpu_only | strong_ref | f32_only |
| high | `nitrix.metrics.cross_entropy_with_logits` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.dice` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.dino_cross_entropy` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.focal_loss` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.ibot_cross_entropy` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.info_nce` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.jaccard` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.joint_histogram` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.koleo` | unmeasured | none | unmeasured |
| high | `nitrix.metrics.lncc` | gpu_only | strong_ref | f32_only |
| high | `nitrix.metrics.mutual_information` | gpu_only | strong_ref | f32_only |
| high | `nitrix.metrics.ncc` | gpu_only | strong_ref | f32_only |
| high | `nitrix.metrics.ssd` | gpu_only | strong_ref | f32_only |
| high | `nitrix.morphology.connected_components` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.distance_transform_edt` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.largest_connected_component` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_pool_with_indices_nd` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_unpool_nd` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_decompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_recompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.crop_to_multiple` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.demean` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.euler` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.fixed_point_solve` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.instance_norm` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.l2_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.lp_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.nonzero_bounding_box` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.odeint` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.overlap_add` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.pad_to_multiple` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.percentile_rescale` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.rk4` | unmeasured | none | unmeasured |
| high | `nitrix.register.bending_energy` | unmeasured | none | unmeasured |
| high | `nitrix.register.gradient_smoothness` | unmeasured | none | unmeasured |
| high | `nitrix.register.jacobian_folding_penalty` | unmeasured | none | unmeasured |
| high | `nitrix.semiring.ell_row_softmax` | unmeasured | none | unmeasured |
| high | `nitrix.semiring.semiring_conv` | unmeasured | none | unmeasured |
| high | `nitrix.semiring.semiring_ell_matmul` | unmeasured | none | unmeasured |
| high | `nitrix.signal.bandpass` | unmeasured | none | unmeasured |
| high | `nitrix.signal.bandstop` | unmeasured | none | unmeasured |
| high | `nitrix.signal.highpass` | unmeasured | none | unmeasured |
| high | `nitrix.signal.iir_filter` | unmeasured | none | unmeasured |
| high | `nitrix.signal.linear_interpolate` | unmeasured | none | unmeasured |
| high | `nitrix.signal.lowpass` | unmeasured | none | unmeasured |
| high | `nitrix.signal.sample_windows` | unmeasured | none | unmeasured |
| high | `nitrix.smoothing.brute_force_knn` | unmeasured | none | unmeasured |
| high | `nitrix.smoothing.susan_emulator` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.compute_vertex_normals` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_add_self_loops` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_mask` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_pad` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_to_dense` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_bary_upsample` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_coarsen_meanpool` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_laplacian_smooth` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_pool_max` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_unpool_max` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.sectioned_semiring_ell_matmul` | unmeasured | none | unmeasured |
| high | `nitrix.stats.env_inst` | unmeasured | none | unmeasured |
| high | `nitrix.stats.gaussian_nll` | unmeasured | none | unmeasured |
| high | `nitrix.stats.instantaneous_frequency` | unmeasured | none | unmeasured |
| high | `nitrix.stats.instantaneous_phase` | unmeasured | none | unmeasured |
| high | `nitrix.stats.kl_diagonal_gaussian` | unmeasured | none | unmeasured |
| high | `nitrix.stats.lme.flame_two_level` | cpu_only | floor_only | f32_only |
| high | `nitrix.stats.product_filter` | unmeasured | none | unmeasured |
| high | `nitrix.stats.product_filtfilt` | unmeasured | none | unmeasured |
| medium | `nitrix.bias.histogram_match` | multiplatform | none | f32_only |
| medium | `nitrix.bias.n4_bias_field_correction` | multiplatform | none | f32_only |
| medium | `nitrix.linalg.tangent_project_spd` | multiplatform | floor_only | f32_only |
| medium | `nitrix.register.affine_register` | multiplatform | none | f32_only |
| medium | `nitrix.register.diffeomorphic_demons_register` | multiplatform | none | f32_only |
| medium | `nitrix.register.rigid_register` | multiplatform | none | f32_only |
| medium | `nitrix.semiring.semiring_ell_edge_aggregate` | multiplatform | none | f32_only |
| medium | `nitrix.semiring.semiring_matmul` | multiplatform | internal_only | f32_only |
| medium | `nitrix.smoothing.bilateral_gaussian` | multiplatform | none | f32_only |
| medium | `nitrix.stats.lme.reml_fit` | multiplatform | none | f32_only |

## Covered with a strong GPU ref — nitrix ahead

| op | strong GPU ref | ratio (ref/nitrix) | nitrix |
|---|---|---:|---|
| `nitrix.signal.lomb_scargle_periodogram` | cupyx.scipy.signal.lombscargle | 109 | ~108.7x faster |
| `nitrix.stats.cov` | cupy.cov | 29.1 | ~29.1x faster |
| `nitrix.stats.corr` | cupy.corrcoef | 27.9 | ~27.9x faster |
| `nitrix.numerics.robust_zscore_normalize` | cupy.robust_zscore_normalize | 12.8 | ~12.8x faster |
| `nitrix.geometry.integrate_velocity_field` | cupy.integrate_velocity_field | 12.8 | ~12.8x faster |
| `nitrix.geometry.spherical_geodesic_distance` | cupy.spherical_geodesic_distance | 11.4 | ~11.4x faster |
| `nitrix.signal.polynomial_detrend` | cupy.lstsq_detrend | 11.3 | ~11.3x faster |
| `nitrix.geometry.spherical_conv` | cupy.spherical_conv | 8.49 | ~8.5x faster |
| `nitrix.signal.sosfiltfilt` | cupyx.scipy.signal.sosfiltfilt | 7.37 | ~7.4x faster |
| `nitrix.geometry.jacobian_det_displacement` | cupy.jacobian_det_displacement | 7.07 | ~7.1x faster |
| `nitrix.geometry.jacobian_displacement` | cupy.jacobian_displacement | 7.05 | ~7.0x faster |
| `nitrix.linalg.residualise` | cupy.linalg.lstsq | 7.04 | ~7.0x faster |
| `nitrix.metrics.mutual_information` | cupy.mi | 5.84 | ~5.8x faster |
| `nitrix.geometry.compactness_penalty` | cupy.compactness_penalty | 5.09 | ~5.1x faster |
| `nitrix.geometry.resample` | cupyx.scipy.ndimage.map_coordinates | 4.18 | ~4.2x faster |
| `nitrix.geometry.displacement_from_reference_grid` | cupy.displacement_from_reference_grid | 3.39 | ~3.4x faster |
| `nitrix.linalg.gaussian_kernel` | cupy.gaussian_kernel | 3.24 | ~3.2x faster |
| `nitrix.linalg.rbf_kernel` | cupy.rbf_kernel | 3.2 | ~3.2x faster |
| `nitrix.metrics.correlation_ratio` | cupy.cr | 3 | ~3.0x faster |
| `nitrix.metrics.lncc` | cupy.lncc | 2.77 | ~2.8x faster |
| `nitrix.geometry.center_of_mass_grid` | cupy.center_of_mass_grid | 2.63 | ~2.6x faster |
| `nitrix.linalg.linear_distance` | cupy.linear_distance | 2.58 | ~2.6x faster |
| `nitrix.morphology.close` | cupyx.scipy.ndimage.grey_closing | 2.49 | ~2.5x faster |
| `nitrix.geometry.sphere_grid_pad_2d` | cupy.sphere_grid_pad_2d | 2.4 | ~2.4x faster |
| `nitrix.stats.partialcorr` | cupy.partialcorr | 2.33 | ~2.3x faster |
| `nitrix.stats.partialcov` | cupy.partialcov | 2.29 | ~2.3x faster |
| `nitrix.stats.precision` | cupy.inv_cov | 2.23 | ~2.2x faster |
| `nitrix.morphology.open` | cupyx.scipy.ndimage.grey_opening | 2.23 | ~2.2x faster |
| `nitrix.smoothing.gaussian` | cupyx.scipy.ndimage.gaussian_filter | 2.2 | ~2.2x faster |
| `nitrix.stats.conditionalcorr` | cupy.conditionalcorr | 2.18 | ~2.2x faster |
| `nitrix.signal.sosfilt` | cupyx.scipy.signal.sosfilt | 2.16 | ~2.2x faster |
| `nitrix.numerics.zscore_normalize` | cupy.zscore_normalize | 2.05 | ~2.0x faster |
| `nitrix.metrics.ncc` | cupy.ncc | 2.02 | ~2.0x faster |
| `nitrix.stats.conditionalcov` | cupy.conditionalcov | 1.99 | ~2.0x faster |
| `nitrix.stats.pca_inverse_transform` | cupy.matmul | 1.93 | ~1.9x faster |
| `nitrix.numerics.psc_normalize` | cupy.psc_normalize | 1.87 | ~1.9x faster |
| `nitrix.signal.tsconv` | cupyx.scipy.signal.correlate | 1.77 | ~1.8x faster |
| `nitrix.stats.pairedcorr` | cupy.pairedcorr | 1.74 | ~1.7x faster |
| `nitrix.geometry.cartesian_to_latlong` | cupy.cartesian_to_latlong | 1.74 | ~1.7x faster |
| `nitrix.linalg.polynomial_kernel` | cupy.polynomial_kernel | 1.72 | ~1.7x faster |
| `nitrix.graph.relaxed_modularity` | cupy.relaxed_modularity | 1.56 | ~1.6x faster |
| `nitrix.linalg.sigmoid_kernel` | cupy.sigmoid_kernel | 1.48 | ~1.5x faster |
| `nitrix.geometry.latlong_to_cartesian` | cupy.latlong_to_cartesian | 1.46 | ~1.5x faster |
| `nitrix.morphology.dilate` | cupyx.scipy.ndimage.grey_dilation | 1.39 | ~1.4x faster |
| `nitrix.stats.pairedcov` | cupy.pairedcov | 1.39 | ~1.4x faster |
| `nitrix.linalg.cosine_kernel` | cupy.cosine_kernel | 1.34 | ~1.3x faster |
| `nitrix.geometry.displacement_from_reference_points` | cupy.displacement_from_reference_points | 1.33 | ~1.3x faster |
| `nitrix.morphology.erode` | cupyx.scipy.ndimage.grey_erosion | 1.2 | ~1.2x faster |
| `nitrix.graph.modularity_matrix` | cupy.modularity_matrix | 1.18 | ~1.2x faster |
| `nitrix.graph.girvan_newman_null` | cupy.gn_null | 1.16 | ~1.2x faster |
| `nitrix.stats.envelope` | cupyx.scipy.signal.hilbert | 1.13 | ~1.1x faster |
| `nitrix.stats.analytic_signal` | cupyx.scipy.signal.hilbert | 1.11 | ~1.1x faster |
| `nitrix.signal.lomb_scargle_interpolate` | cupy.joint_glm | 1.1 | ~1.1x faster |
| `nitrix.stats.hilbert_transform` | cupyx.scipy.signal.hilbert | 1.03 | ~1.0x faster |
| `nitrix.graph.coaffiliation` | cupy.coaffiliation | 1.02 | ~1.0x faster |

## Scale — brain-scale tier (COVERAGE v2)

Ops declaring a `large_param_points` tier: did nitrix run at the largest realistic size, or break (oom/timeout) before it? `scale_capped` = **fragility at scale** (the win at a small size may not hold where practitioners run).

| op | scaled to | capped by |
|---|---|---|
| `nitrix.stats.lme.flame_two_level` | 131072 elem | **timeout** |

## Economic — GPU as a multiple of CPU (COVERAGE v2)

The deployment-economics bar: a nitrix-GPU win counts only if it is **multiplicative** over the CPU gold standard (the GPU hardware premium; see `ECONOMIC.md`). Verdict at largest real/large point, else representative (`~` = not authoritative). `not multiplicative enough` is a real GPU win that still fails the cost test.

| op | verdict | amortized | domain ref |
|---|---|---:|---|
| `nitrix.signal.tsconv` | not multiplicative enough ~ | 3.5x | — |
| `nitrix.geometry.latlong_to_cartesian` | not multiplicative enough ~ | 2.1x | — |
| `nitrix.geometry.center_of_mass_points` | not multiplicative enough ~ | 1.5x | — |
| `nitrix.geometry.displacement_from_reference_points` | not multiplicative enough ~ | 1.3x | — |
| `nitrix.graph.coaffiliation` | not multiplicative enough ~ | 1.2x | — |
| `nitrix.graph.girvan_newman_null` | not multiplicative enough ~ | 0.7x | — |
| `nitrix.graph.laplacian` | not multiplicative enough ~ | 0.5x | — |
| `nitrix.geometry.sphere_grid_pad_2d` | not multiplicative enough ~ | 0.3x | — |
| `nitrix.graph.degree_vector` | not multiplicative enough ~ | 0.2x | — |
| `nitrix.geometry.sphere_grid_unpad_2d` | not multiplicative enough ~ | 0.0x | — |
| `nitrix.signal.lomb_scargle_periodogram` | favorable (amortized only) ~ | 1386.7x | — |
| `nitrix.signal.polynomial_detrend` | favorable (amortized only) ~ | 469.0x | — |
| `nitrix.linalg.polynomial_kernel` | favorable (amortized only) ~ | 337.8x | — |
| `nitrix.stats.pca_inverse_transform` | favorable (amortized only) | 261.5x | — |
| `nitrix.stats.lme.reml_fit` | favorable (amortized only) ~ | 257.1x | statsmodels.MixedLM |
| `nitrix.linalg.residualise` | favorable (amortized only) ~ | 205.9x | — |
| `nitrix.bias.n4_bias_field_correction` | favorable (amortized only) ~ | 190.1x | simpleitk.N4 |
| `nitrix.linalg.rbf_kernel` | favorable (amortized only) ~ | 179.1x | — |
| `nitrix.linalg.gaussian_kernel` | favorable (amortized only) ~ | 176.4x | — |
| `nitrix.linalg.sigmoid_kernel` | favorable (amortized only) ~ | 155.2x | — |
| `nitrix.linalg.symlog` | favorable (amortized only) ~ | 128.7x | — |
| `nitrix.linalg.linear_distance` | favorable (amortized only) ~ | 116.4x | — |
| `nitrix.graph.modularity_matrix` | favorable (amortized only) ~ | 116.2x | — |
| `nitrix.morphology.distance_transform` | favorable (amortized only) ~ | 116.1x | simpleitk.DanielssonDistanceMap |
| `nitrix.geometry.spherical_conv` | favorable (amortized only) ~ | 110.6x | — |
| `nitrix.geometry.integrate_velocity_field` | favorable (amortized only) ~ | 102.8x | — |
| `nitrix.linalg.sympower` | favorable (amortized only) ~ | 102.4x | — |
| `nitrix.numerics.zscore_normalize` | favorable (amortized only) ~ | 98.2x | — |
| `nitrix.stats.cov` | favorable (amortized only) ~ | 95.1x | — |
| `nitrix.geometry.spherical_geodesic_distance` | favorable (amortized only) ~ | 93.2x | — |
| `nitrix.stats.corr` | favorable (amortized only) ~ | 91.1x | — |
| `nitrix.graph.relaxed_modularity` | favorable (amortized only) ~ | 87.6x | — |
| `nitrix.stats.conditionalcorr` | favorable (amortized only) | 84.1x | — |
| `nitrix.stats.pairedcov` | favorable (amortized only) | 81.1x | — |
| `nitrix.stats.analytic_signal` | favorable (amortized only) ~ | 78.9x | — |
| `nitrix.stats.envelope` | favorable (amortized only) ~ | 76.9x | — |
| `nitrix.numerics.robust_zscore_normalize` | favorable (amortized only) ~ | 73.2x | — |
| `nitrix.stats.hilbert_transform` | favorable (amortized only) ~ | 72.3x | — |
| `nitrix.stats.conditionalcov` | favorable (amortized only) | 70.9x | — |
| `nitrix.linalg.linear_kernel` | favorable (amortized only) ~ | 70.4x | — |
| `nitrix.numerics.intensity_normalize` | favorable (amortized only) ~ | 70.1x | — |
| `nitrix.linalg.cosine_kernel` | favorable (amortized only) ~ | 68.8x | — |
| `nitrix.semiring.semiring_ell_edge_aggregate` | favorable (amortized only) ~ | 68.3x | — |
| `nitrix.stats.pca_transform` | favorable (amortized only) | 65.7x | — |
| `nitrix.geometry.compactness_penalty` | favorable (amortized only) ~ | 65.2x | — |
| `nitrix.numerics.psc_normalize` | favorable (amortized only) ~ | 54.4x | — |
| `nitrix.bias.histogram_match` | favorable (amortized only) ~ | 43.6x | simpleitk.HistogramMatching |
| `nitrix.linalg.symexp` | favorable (amortized only) ~ | 34.0x | — |
| `nitrix.signal.sosfiltfilt` | favorable (amortized only) ~ | 33.2x | — |
| `nitrix.stats.pca_fit` | favorable (amortized only) | 30.3x | — |
| `nitrix.signal.sosfilt` | favorable (amortized only) ~ | 28.9x | — |
| `nitrix.smoothing.gaussian` | favorable (amortized only) ~ | 28.0x | — |
| `nitrix.stats.pairedcorr` | favorable (amortized only) | 27.3x | — |
| `nitrix.morphology.close` | favorable (amortized only) ~ | 23.5x | — |
| `nitrix.linalg.tangent_project_spd` | favorable (amortized only) ~ | 23.1x | — |
| `nitrix.register.diffeomorphic_demons_register` | favorable (amortized only) | 23.1x | ants.registration |
| `nitrix.morphology.open` | favorable (amortized only) ~ | 22.3x | — |
| `nitrix.semiring.semiring_matmul` | favorable (amortized only) ~ | 21.5x | — |
| `nitrix.geometry.jacobian_displacement` | favorable (amortized only) ~ | 21.4x | — |
| `nitrix.geometry.jacobian_det_displacement` | favorable (amortized only) ~ | 21.2x | — |
| `nitrix.geometry.spatial_transform` | favorable (amortized only) ~ | 19.6x | — |
| `nitrix.signal.lomb_scargle_interpolate` | favorable (amortized only) ~ | 18.9x | — |
| `nitrix.morphology.dilate` | favorable (amortized only) ~ | 13.3x | simpleitk.GrayscaleDilate |
| `nitrix.morphology.erode` | favorable (amortized only) ~ | 12.1x | simpleitk.GrayscaleErode |
| `nitrix.smoothing.bilateral_gaussian` | favorable (amortized only) ~ | 10.8x | simpleitk.Bilateral |
| `nitrix.linalg.symsqrt` | favorable (amortized only) ~ | 10.2x | — |
| `nitrix.register.rigid_register` | favorable (amortized only) | 8.7x | ants.registration |
| `nitrix.geometry.cartesian_to_latlong` | favorable (amortized only) ~ | 8.5x | — |
| `nitrix.stats.partialcorr` | favorable (amortized only) ~ | 7.8x | — |
| `nitrix.morphology.median_filter` | favorable (amortized only) ~ | 7.7x | simpleitk.Median |
| `nitrix.geometry.displacement_from_reference_grid` | favorable (amortized only) ~ | 7.6x | — |
| `nitrix.register.affine_register` | favorable (amortized only) | 7.4x | ants.registration |
| `nitrix.geometry.center_of_mass_grid` | favorable (amortized only) ~ | 6.9x | — |
| `nitrix.stats.precision` | favorable (amortized only) ~ | 6.7x | — |
| `nitrix.stats.partialcov` | favorable (amortized only) ~ | 6.6x | — |

## Real-data coverage (COVERAGE v2)

Marquee functions should be tested on real brain data against real community baselines. Realism ladder: `synthetic` < `real_planted` (real image, planted/known truth) < `real_full` (actual problem). Which ops are *required* to reach real data is tier-gated -- see the marquee matrix below.

| op | realism | domain ref (on) |
|---|---|---|
| `nitrix.register.rigid_register` | real_planted | ants.registration (real_planted) |
| `nitrix.register.affine_register` | real_planted | ants.registration (real_planted) |
| `nitrix.register.diffeomorphic_demons_register` | real_planted | ants.registration (real_planted) |

## Marquee coverage matrix (COVERAGE v2)

The headline functions used on real images, scored against their tier bar (`score` = satisfied / applicable required axes). Glyphs: `✓` met · `✗` unmet · `⚠` fragile · `~` non-authoritative · `·` n/a. Worst-covered first.

| op | score | platform | scale | economic | input | gpu-ref | domain-ref |
|---|---|---|---|---|---|---|---|
| `nitrix.stats.lme.flame_two_level` | 1/5 | ✗ cpu_only | ⚠ timeout | · | ✗ synth | · | ◐ fsl.flameo |
| `nitrix.stats.lme.reml_fit` | 2/5 | ✓ | ○ | ✓~ | ✗ synth | · | ◐ statsmodels.MixedLM |
| `nitrix.bias.n4_bias_field_correction` | 2/4 | ✓ | · | ✓~ | ✗ synth | · | ◐ simpleitk.N4 |
| `nitrix.numerics.intensity_normalize` | 2/4 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.smoothing.bilateral_gaussian` | 2/4 | ✓ | · | ✓~ | ✗ synth | · | ◐ simpleitk.Bilateral |
| `nitrix.register.affine_register` | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |
| `nitrix.register.diffeomorphic_demons_register` | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |
| `nitrix.register.rigid_register` | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |

**Marquee unmet** (no real-data input, or no domain ref on real data) — the next-round targets: `flame_two_level`, `reml_fit`, `intensity_normalize`, `bilateral_gaussian`, `n4_bias_field_correction`.

## Full coverage matrix — every op with a case (COVERAGE v2)

All 99 ops with a case, scored against their tier (`★` = marquee, which adds the real-data + domain-on-real bar). Worst-covered (and marquee) first; same glyphs as above.

| op | ★ | score | platform | scale | economic | input | gpu-ref | domain-ref |
|---|---|---|---|---|---|---|---|---|
| `nitrix.stats.lme.flame_two_level` | ★ | 1/5 | ✗ cpu_only | ⚠ timeout | · | ✗ synth | · | ◐ fsl.flameo |
| `nitrix.stats.lme.reml_fit` | ★ | 2/5 | ✓ | ○ | ✓~ | ✗ synth | · | ◐ statsmodels.MixedLM |
| `nitrix.bias.n4_bias_field_correction` | ★ | 2/4 | ✓ | · | ✓~ | ✗ synth | · | ◐ simpleitk.N4 |
| `nitrix.numerics.intensity_normalize` | ★ | 2/4 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.smoothing.bilateral_gaussian` | ★ | 2/4 | ✓ | · | ✓~ | ✗ synth | · | ◐ simpleitk.Bilateral |
| `nitrix.geometry.affine_exp` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.geometry.compose_velocity` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.geometry.invert_displacement` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.geometry.rigid_exp` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.geometry.rigid_log` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.geometry.spatial_gradient` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.linalg.matrix_exp` |  | 1/3 | ✗ gpu_only | ✓ | ✗~ | ✗ synth | · | ✗ none |
| `nitrix.morphology.connected_components` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.morphology.distance_transform_edt` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.morphology.largest_connected_component` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.morphology.max_pool_with_indices_nd` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.morphology.max_unpool_nd` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.register.bending_energy` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.register.gradient_smoothness` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.register.jacobian_folding_penalty` |  | 0/2 | ✗ unmeasured | · | · | ✗ synth | · | ✗ none |
| `nitrix.linalg.tangent_project_spd` |  | 1/2 | ✓ | · | ✓~ | ✗ synth | · | ✗ none |
| `nitrix.metrics.correlation_ratio` |  | 1/2 | ✗ gpu_only | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.metrics.lncc` |  | 1/2 | ✗ gpu_only | · | ✗~ | ✗ synth | ✓ | ◐ simpleitk.ANTSNeighborhoodCorrelation |
| `nitrix.metrics.mutual_information` |  | 1/2 | ✗ gpu_only | · | ✗~ | ✗ synth | ✓ | ◐ simpleitk.MattesMI |
| `nitrix.metrics.ncc` |  | 1/2 | ✗ gpu_only | · | ✗~ | ✗ synth | ✓ | ◐ simpleitk.Correlation |
| `nitrix.metrics.ssd` |  | 1/2 | ✗ gpu_only | · | ✗~ | ✗ synth | ✓ | ◐ simpleitk.MeanSquares |
| `nitrix.semiring.semiring_ell_edge_aggregate` |  | 1/2 | ✓ | · | ✓~ | ✗ synth | · | ✗ none |
| `nitrix.semiring.semiring_matmul` |  | 1/2 | ✓ | · | ✓~ | ✗ synth | · | ✗ none |
| `nitrix.stats.corr` |  | 2/3 | ✓ | ○ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.cov` |  | 2/3 | ✓ | ○ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.partialcorr` |  | 2/3 | ✓ | ○ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.partialcov` |  | 2/3 | ✓ | ○ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.precision` |  | 2/3 | ✓ | ○ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.register.affine_register` | ★ | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |
| `nitrix.register.diffeomorphic_demons_register` | ★ | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |
| `nitrix.register.rigid_register` | ★ | 5/5 | ✓ | ✓ | ✓ | ◐ planted | · | ● ants.registration |
| `nitrix.bias.histogram_match` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | · | ◐ simpleitk.HistogramMatching |
| `nitrix.geometry.cartesian_to_latlong` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.center_of_mass_grid` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.center_of_mass_points` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.compactness_penalty` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.displacement_from_reference_grid` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.displacement_from_reference_points` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.integrate_velocity_field` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.jacobian_det_displacement` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.jacobian_displacement` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.latlong_to_cartesian` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.resample` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ◐ ants.resample_image |
| `nitrix.geometry.spatial_transform` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.sphere_grid_pad_2d` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.sphere_grid_unpad_2d` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.spherical_conv` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.geometry.spherical_geodesic_distance` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.coaffiliation` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.degree_vector` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.diffusion_embedding` |  | 3/3 | ✓ | ✓ | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.girvan_newman_null` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.laplacian` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.laplacian_eigenmap` |  | 3/3 | ✓ | ✓ | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.modularity_matrix` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.graph.relaxed_modularity` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.cosine_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.gaussian_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.linear_distance` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.linear_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.polynomial_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.rbf_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.residualise` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.sigmoid_kernel` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.symexp` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.symlog` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.sympower` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.linalg.symsqrt` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.morphology.close` |  | 3/3 | ✓ | ✓ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.morphology.dilate` |  | 3/3 | ✓ | ✓ | ✓~ | ✗ synth | ✓ | ◐ simpleitk.GrayscaleDilate |
| `nitrix.morphology.distance_transform` |  | 3/3 | ✓ | ✓ | ✓~ | ✗ synth | ✓ | ◐ simpleitk.DanielssonDistanceMap |
| `nitrix.morphology.erode` |  | 3/3 | ✓ | ✓ | ✓~ | ✗ synth | ✓ | ◐ simpleitk.GrayscaleErode |
| `nitrix.morphology.median_filter` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ◐ simpleitk.Median |
| `nitrix.morphology.open` |  | 3/3 | ✓ | ✓ | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.numerics.psc_normalize` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.numerics.robust_zscore_normalize` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.numerics.zscore_normalize` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.lomb_scargle_interpolate` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.lomb_scargle_periodogram` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.polynomial_detrend` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.sosfilt` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.sosfiltfilt` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.signal.tsconv` |  | 2/2 | ✓ | · | ✗~ | ✗ synth | ✓ | ✗ none |
| `nitrix.smoothing.gaussian` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.analytic_signal` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.conditionalcorr` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.conditionalcov` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.envelope` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.hilbert_transform` |  | 2/2 | ✓ | · | ✓~ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.pairedcorr` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.pairedcov` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.pca_fit` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.pca_inverse_transform` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |
| `nitrix.stats.pca_transform` |  | 3/3 | ✓ | ✓ | ✓ | ✗ synth | ✓ | ✗ none |

## Caveats

- `ratio = strong_ref.min / nitrix.min` at the op's representative point; `<1` ⇒ nitrix slower. The "≈Nx" column is its reciprocal (presentation only).
- A **provisional** op's latest data came from a `--skip-slow` (fast) run; run the full sweep before acting (mandate §7).
- "Lagging" is currently *slower than the strong on-target ref*; per-op **targets** (mandate §2.4) will refine the bar.
- Host-side constructors (jit `n/a`) are excluded from the runtime denominator; they have no device-time bar.

