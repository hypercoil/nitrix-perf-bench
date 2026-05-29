# nitrix-perf-bench — coverage & deficit report

> Generated from the L4 store joined with the nitrix op catalogue (`op_matrix.json`). No values are hand-edited; every ratio is read from the stored rows (SCHEMA §G).

## Coverage (runtime ops)

- **runtime ops catalogued**: 52 (+ 7 host-side constructors, apart)
- **measured** (≥1 platform): 14 / 52
- **multiplatform** (CPU + GPU): 14 / 52
- **with a strong on-target GPU ref**: 12 / 52
- **lagging on the GPU**: 5
- **GPU blocked upstream** (jaxlib cuSOLVER): 0

## Lagging on the deployment target (GPU) — ranked

nitrix is slower than its strong on-target reference here (`ratio = ref/nitrix < 1`); worst first. The Pallas-kernel / algorithm candidates.

| # | op | strong GPU ref | ratio (ref/nitrix) | nitrix | note |
|---|---|---|---:|---|---|
| 1 | `nitrix.morphology.distance_transform` | cupyx.scipy.ndimage.distance_transform_edt | 0.00954 | ~104.8x slower |  |
| 2 | `nitrix.morphology.median_filter` | cupyx.scipy.ndimage.median_filter | 0.191 | ~5.2x slower |  |
| 3 | `nitrix.geometry.spatial_transform` | cupyx.scipy.ndimage.map_coordinates | 0.557 | ~1.8x slower |  |
| 4 | `nitrix.morphology.erode` | cupyx.scipy.ndimage.grey_erosion | 0.777 | ~1.3x slower |  |
| 5 | `nitrix.morphology.dilate` | cupyx.scipy.ndimage.grey_dilation | 0.825 | ~1.2x slower |  |

## Under-covered — ranked by priority

Priority is a coarse heuristic (no consumer-traffic weighting yet): **high** = unmeasured or missing a platform; **medium** = measured on both but no strong on-target GPU ref (no apples-to-apples bar).

| priority | op | coverage | ref strength | precision |
|---|---|---|---|---|
| high | `nitrix.geometry.integrate_velocity_field` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.jacobian_det_displacement` | unmeasured | none | unmeasured |
| high | `nitrix.geometry.sphere_grid_pad_2d` | unmeasured | none | unmeasured |
| high | `nitrix.graph.degree_vector` | unmeasured | none | unmeasured |
| high | `nitrix.graph.laplacian` | unmeasured | none | unmeasured |
| high | `nitrix.graph.laplacian_eigenmap` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.linear_distance` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.linear_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.mean_log_euclidean` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.rbf_kernel` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.recondition_eigenspaces` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.sym2vec` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.symmetric` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.toeplitz_2d` | unmeasured | none | unmeasured |
| high | `nitrix.linalg.vec2sym` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_pool_with_indices_nd` | unmeasured | none | unmeasured |
| high | `nitrix.morphology.max_unpool_nd` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.complex_decompose` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.intensity_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.numerics.zscore_normalize` | unmeasured | none | unmeasured |
| high | `nitrix.semiring.semiring_conv` | unmeasured | none | unmeasured |
| high | `nitrix.semiring.semiring_ell_matmul` | unmeasured | none | unmeasured |
| high | `nitrix.signal.linear_interpolate` | unmeasured | none | unmeasured |
| high | `nitrix.signal.lomb_scargle_interpolate` | unmeasured | none | unmeasured |
| high | `nitrix.signal.lomb_scargle_periodogram` | unmeasured | none | unmeasured |
| high | `nitrix.signal.polynomial_detrend` | unmeasured | none | unmeasured |
| high | `nitrix.signal.tsconv` | unmeasured | none | unmeasured |
| high | `nitrix.smoothing.bilateral_gaussian` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_bary_upsample` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_pool_max` | unmeasured | none | unmeasured |
| high | `nitrix.sparse.mesh_unpool_max` | unmeasured | none | unmeasured |
| high | `nitrix.stats.analytic_signal` | unmeasured | none | unmeasured |
| high | `nitrix.stats.envelope` | unmeasured | none | unmeasured |
| high | `nitrix.stats.hilbert_transform` | unmeasured | none | unmeasured |
| high | `nitrix.stats.lme.flame_two_level` | unmeasured | none | unmeasured |
| high | `nitrix.stats.lme.reml_fit` | unmeasured | none | unmeasured |
| high | `nitrix.stats.partialcov` | unmeasured | none | unmeasured |
| high | `nitrix.stats.precision` | unmeasured | none | unmeasured |
| medium | `nitrix.semiring.semiring_ell_edge_aggregate` | multiplatform | none | f32_only |
| medium | `nitrix.semiring.semiring_matmul` | multiplatform | internal_only | f32_only |

## Covered with a strong GPU ref — nitrix ahead

| op | strong GPU ref | ratio (ref/nitrix) | nitrix |
|---|---|---:|---|
| `nitrix.stats.cov` | cupy.cov | 29.1 | ~29.1x faster |
| `nitrix.stats.corr` | cupy.corrcoef | 27.9 | ~27.9x faster |
| `nitrix.linalg.residualise` | cupy.linalg.lstsq | 6.81 | ~6.8x faster |
| `nitrix.smoothing.gaussian` | cupyx.scipy.ndimage.gaussian_filter | 2.2 | ~2.2x faster |

## Caveats

- `ratio = strong_ref.min / nitrix.min` at the op's representative point; `<1` ⇒ nitrix slower. The "≈Nx" column is its reciprocal (presentation only).
- A **provisional** op's latest data came from a `--skip-slow` (fast) run; run the full sweep before acting (mandate §7).
- "Lagging" is currently *slower than the strong on-target ref*; per-op **targets** (mandate §2.4) will refine the bar.
- Host-side constructors (jit `n/a`) are excluded from the runtime denominator; they have no device-time bar.

