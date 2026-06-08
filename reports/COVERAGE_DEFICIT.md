# nitrix-perf-bench — coverage & deficit report

> Generated from the L4 store joined with the nitrix op catalogue (`op_matrix.json`). No values are hand-edited; every ratio is read from the stored rows (SCHEMA §G).

## Coverage (runtime ops)

- **runtime ops catalogued**: 128 (+ 15 host-side constructors, apart)
- **measured** (≥1 platform): 69 / 128
- **multiplatform** (CPU + GPU): 68 / 128
- **with a strong on-target GPU ref**: 61 / 128
- **lagging on the GPU**: 9
- **GPU blocked upstream** (jaxlib cuSOLVER): 2

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
| 7 | `nitrix.geometry.center_of_mass_points` | cupy.center_of_mass_points | 0.698 | ~1.4x slower |  |
| 8 | `nitrix.graph.laplacian` | cupy.laplacian | 0.743 | ~1.3x slower |  |
| 9 | `nitrix.morphology.distance_transform` | cupyx.scipy.ndimage.distance_transform_edt | 0.963 | ~1.0x slower | provisional (fast run) |

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
| high | `nitrix.bias.bias_field_correction` | unmeasured | none | unmeasured |
| high | `nitrix.bias.bspline_approximate` | unmeasured | none | unmeasured |
| high | `nitrix.bias.sharpen_histogram` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[cubic]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[lanczos]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[multilabel]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample[nearest]` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.spatial_transform[lanczos]` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cone_project_spd` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.delete_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.fill_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_log_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.parameterised_norm` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.recondition_eigenspaces` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.squareform` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.sym2vec` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmap` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmetric` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz_2d` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.vec2sym` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.distance_transform_edt` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_pool_with_indices_nd` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_unpool_nd` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_decompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_recompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.demean` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.percentile_rescale` | unmeasured | none | unmeasured |
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
| high | `nitrix.sparse.ell_add_self_loops` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_mask` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_pad` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.ell_to_dense` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_bary_upsample` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_coarsen_meanpool` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_pool_max` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_unpool_max` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.sectioned_semiring_ell_matmul` | unmeasured | none | unmeasured |
| high | `nitrix.stats.conditionalcorr` | unmeasured | none | unmeasured |
| high | `nitrix.stats.conditionalcov` | unmeasured | none | unmeasured |
| high | `nitrix.stats.env_inst` | unmeasured | none | unmeasured |
| high | `nitrix.stats.instantaneous_frequency` | unmeasured | none | unmeasured |
| high | `nitrix.stats.instantaneous_phase` | unmeasured | none | unmeasured |
| high | `nitrix.stats.lme.flame_two_level` | cpu_only | none | f32_only |
| high | `nitrix.stats.pairedcorr` | unmeasured | none | unmeasured |
| high | `nitrix.stats.pairedcov` | unmeasured | none | unmeasured |
| high | `nitrix.stats.product_filter` | unmeasured | none | unmeasured |
| high | `nitrix.stats.product_filtfilt` | unmeasured | none | unmeasured |
| medium | `nitrix.bias.histogram_match` | multiplatform | floor_only | f32_only |
| medium | `nitrix.bias.n4_bias_field_correction` | multiplatform | floor_only | f32_only |
| medium | `nitrix.linalg.tangent_project_spd` | multiplatform | floor_only | f32_only |
| medium | `nitrix.semiring.semiring_ell_edge_aggregate` | multiplatform | none | f32_only |
| medium | `nitrix.semiring.semiring_matmul` | multiplatform | internal_only | f32_only |
| medium | `nitrix.smoothing.bilateral_gaussian` | multiplatform | floor_only | f32_only |
| medium | `nitrix.stats.lme.reml_fit` | multiplatform | floor_only | f32_only |

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
| `nitrix.geometry.compactness_penalty` | cupy.compactness_penalty | 5.09 | ~5.1x faster |
| `nitrix.geometry.resample` | cupyx.scipy.ndimage.map_coordinates | 4.9 | ~4.9x faster |
| `nitrix.geometry.displacement_from_reference_grid` | cupy.displacement_from_reference_grid | 3.39 | ~3.4x faster |
| `nitrix.linalg.gaussian_kernel` | cupy.gaussian_kernel | 3.24 | ~3.2x faster |
| `nitrix.linalg.rbf_kernel` | cupy.rbf_kernel | 3.2 | ~3.2x faster |
| `nitrix.geometry.center_of_mass_grid` | cupy.center_of_mass_grid | 2.63 | ~2.6x faster |
| `nitrix.linalg.linear_distance` | cupy.linear_distance | 2.58 | ~2.6x faster |
| `nitrix.morphology.close` | cupyx.scipy.ndimage.grey_closing | 2.49 | ~2.5x faster |
| `nitrix.geometry.sphere_grid_pad_2d` | cupy.sphere_grid_pad_2d | 2.4 | ~2.4x faster |
| `nitrix.stats.partialcorr` | cupy.partialcorr | 2.33 | ~2.3x faster |
| `nitrix.stats.partialcov` | cupy.partialcov | 2.29 | ~2.3x faster |
| `nitrix.stats.precision` | cupy.inv_cov | 2.23 | ~2.2x faster |
| `nitrix.morphology.open` | cupyx.scipy.ndimage.grey_opening | 2.23 | ~2.2x faster |
| `nitrix.smoothing.gaussian` | cupyx.scipy.ndimage.gaussian_filter | 2.2 | ~2.2x faster |
| `nitrix.signal.sosfilt` | cupyx.scipy.signal.sosfilt | 2.16 | ~2.2x faster |
| `nitrix.numerics.zscore_normalize` | cupy.zscore_normalize | 2.05 | ~2.0x faster |
| `nitrix.numerics.psc_normalize` | cupy.psc_normalize | 1.87 | ~1.9x faster |
| `nitrix.signal.tsconv` | cupyx.scipy.signal.correlate | 1.77 | ~1.8x faster |
| `nitrix.geometry.cartesian_to_latlong` | cupy.cartesian_to_latlong | 1.74 | ~1.7x faster |
| `nitrix.linalg.polynomial_kernel` | cupy.polynomial_kernel | 1.72 | ~1.7x faster |
| `nitrix.graph.relaxed_modularity` | cupy.relaxed_modularity | 1.56 | ~1.6x faster |
| `nitrix.linalg.sigmoid_kernel` | cupy.sigmoid_kernel | 1.48 | ~1.5x faster |
| `nitrix.geometry.latlong_to_cartesian` | cupy.latlong_to_cartesian | 1.46 | ~1.5x faster |
| `nitrix.morphology.dilate` | cupyx.scipy.ndimage.grey_dilation | 1.39 | ~1.4x faster |
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

## Caveats

- `ratio = strong_ref.min / nitrix.min` at the op's representative point; `<1` ⇒ nitrix slower. The "≈Nx" column is its reciprocal (presentation only).
- A **provisional** op's latest data came from a `--skip-slow` (fast) run; run the full sweep before acting (mandate §7).
- "Lagging" is currently *slower than the strong on-target ref*; per-op **targets** (mandate §2.4) will refine the bar.
- Host-side constructors (jit `n/a`) are excluded from the runtime denominator; they have no device-time bar.

