# nitrix-perf-bench — coverage & deficit report

> Generated from the L4 store joined with the nitrix op catalogue (`op_matrix.json`). No values are hand-edited; every ratio is read from the stored rows (SCHEMA §G).

## Coverage (runtime ops)

- **runtime ops catalogued**: 122 (+ 15 host-side constructors, apart)
- **measured** (≥1 platform): 32 / 122
- **multiplatform** (CPU + GPU): 31 / 122
- **with a strong on-target GPU ref**: 25 / 122
- **lagging on the GPU**: 6
- **GPU blocked upstream** (jaxlib cuSOLVER): 0

## Lagging on the deployment target (GPU) — ranked

nitrix is slower than its strong on-target reference here (`ratio = ref/nitrix < 1`); worst first. The Pallas-kernel / algorithm candidates.

| # | op | strong GPU ref | ratio (ref/nitrix) | nitrix | note |
|---|---|---|---:|---|---|
| 1 | `nitrix.morphology.distance_transform` | cupyx.scipy.ndimage.distance_transform_edt | 0.00922 | ~108.5x slower |  |
| 2 | `nitrix.morphology.median_filter` | cupyx.scipy.ndimage.median_filter | 0.201 | ~5.0x slower |  |
| 3 | `nitrix.linalg.linear_kernel` | cupy.linear_kernel | 0.518 | ~1.9x slower |  |
| 4 | `nitrix.geometry.spatial_transform` | cupyx.scipy.ndimage.map_coordinates | 0.557 | ~1.8x slower |  |
| 5 | `nitrix.morphology.erode` | cupyx.scipy.ndimage.grey_erosion | 0.765 | ~1.3x slower |  |
| 6 | `nitrix.morphology.dilate` | cupyx.scipy.ndimage.grey_dilation | 0.796 | ~1.3x slower |  |

## Under-covered — ranked by priority

Priority is a coarse heuristic (no consumer-traffic weighting yet): **high** = unmeasured or missing a platform; **medium** = measured on both but no strong on-target GPU ref (no apples-to-apples bar).

| priority | op | coverage | ref strength | precision |
|---|---|---|---|---|
| high | `nitrix.bias.bias_field_correction` | unmeasured | none | unmeasured |
| high | `nitrix.bias.bspline_approximate` | unmeasured | none | unmeasured |
| high | `nitrix.bias.sharpen_histogram` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.cartesian_to_latlong` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.center_of_mass_grid` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.center_of_mass_points` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.compactness_penalty` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.displacement_from_reference_grid` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.displacement_from_reference_points` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.integrate_velocity_field` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.jacobian_det_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.jacobian_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.latlong_to_cartesian` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.resample` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.sphere_grid_pad_2d` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.sphere_grid_unpad_2d` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.spherical_conv` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.spherical_geodesic_distance` | unmeasured | none | unmeasured |
| high | `nitrix.graph.coaffiliation` | unmeasured | none | unmeasured |
| high | `nitrix.graph.degree_vector` | unmeasured | none | unmeasured |
| high | `nitrix.graph.diffusion_embedding` | unmeasured | none | unmeasured |
| high | `nitrix.graph.girvan_newman_null` | unmeasured | none | unmeasured |
| high | `nitrix.graph.laplacian` | unmeasured | none | unmeasured |
| high | `nitrix.graph.laplacian_eigenmap` | unmeasured | none | unmeasured |
| high | `nitrix.graph.modularity_matrix` | unmeasured | none | unmeasured |
| high | `nitrix.graph.relaxed_modularity` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cone_project_spd` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.cosine_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.delete_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.fill_diagonal` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.gaussian_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_log_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.parameterised_norm` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.polynomial_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.recondition_eigenspaces` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.sigmoid_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.squareform` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.sym2vec` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symexp` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmap` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmetric` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz_2d` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.vec2sym` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.close` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_pool_with_indices_nd` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_unpool_nd` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.open` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_decompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_recompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.demean` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.intensity_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.percentile_rescale` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.psc_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.robust_zscore_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.zscore_normalize` | unmeasured | none | unmeasured |
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
| high | `nitrix.signal.sosfilt` | unmeasured | none | unmeasured |
| high | `nitrix.signal.sosfiltfilt` | unmeasured | none | unmeasured |
| high | `nitrix.smoothing.bilateral_gaussian` | unmeasured | none | unmeasured |
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
| medium | `nitrix.stats.lme.reml_fit` | multiplatform | floor_only | f32_only |

## Covered with a strong GPU ref — nitrix ahead

| op | strong GPU ref | ratio (ref/nitrix) | nitrix |
|---|---|---:|---|
| `nitrix.signal.lomb_scargle_periodogram` | cupyx.scipy.signal.lombscargle | 109 | ~108.7x faster |
| `nitrix.stats.cov` | cupy.cov | 29.1 | ~29.1x faster |
| `nitrix.stats.corr` | cupy.corrcoef | 27.9 | ~27.9x faster |
| `nitrix.signal.polynomial_detrend` | cupy.lstsq_detrend | 12.1 | ~12.1x faster |
| `nitrix.linalg.residualise` | cupy.linalg.lstsq | 6.81 | ~6.8x faster |
| `nitrix.linalg.rbf_kernel` | cupy.rbf_kernel | 3.2 | ~3.2x faster |
| `nitrix.linalg.linear_distance` | cupy.linear_distance | 2.58 | ~2.6x faster |
| `nitrix.stats.partialcorr` | cupy.partialcorr | 2.33 | ~2.3x faster |
| `nitrix.stats.partialcov` | cupy.partialcov | 2.29 | ~2.3x faster |
| `nitrix.stats.precision` | cupy.inv_cov | 2.23 | ~2.2x faster |
| `nitrix.smoothing.gaussian` | cupyx.scipy.ndimage.gaussian_filter | 2.2 | ~2.2x faster |
| `nitrix.signal.tsconv` | cupyx.scipy.signal.correlate | 1.77 | ~1.8x faster |
| `nitrix.stats.envelope` | cupyx.scipy.signal.hilbert | 1.13 | ~1.1x faster |
| `nitrix.stats.analytic_signal` | cupyx.scipy.signal.hilbert | 1.11 | ~1.1x faster |
| `nitrix.signal.lomb_scargle_interpolate` | cupy.joint_glm | 1.1 | ~1.1x faster |
| `nitrix.stats.hilbert_transform` | cupyx.scipy.signal.hilbert | 1.03 | ~1.0x faster |

## Caveats

- `ratio = strong_ref.min / nitrix.min` at the op's representative point; `<1` ⇒ nitrix slower. The "≈Nx" column is its reciprocal (presentation only).
- A **provisional** op's latest data came from a `--skip-slow` (fast) run; run the full sweep before acting (mandate §7).
- "Lagging" is currently *slower than the strong on-target ref*; per-op **targets** (mandate §2.4) will refine the bar.
- Host-side constructors (jit `n/a`) are excluded from the runtime denominator; they have no device-time bar.

