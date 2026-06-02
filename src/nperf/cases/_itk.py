# -*- coding: utf-8 -*-
"""Shared helpers for the SimpleITK-parity cases (N4 bias + histogram match).

These two `nitrix.bias` ops were built to match SimpleITK / ITK targets and
nitrix's own test suite asserts live-SimpleITK parity, so SimpleITK is *the*
canonical reference (not a mere floor). The generators + parity criteria here
are copied verbatim from nitrix's parity tests (`tests/test_bias.py`,
`tests/test_histogram_match.py`) so the comparison holds at the exact inputs
nitrix validated.

Neither op has a bit-level fp64 oracle: N4's bias field is defined only up to a
global scale (its parity is correlation + scale-invariant RMSE over the mask),
and histogram matching's truth *is* the ITK landmark algorithm. So the cases
use `fp64_reference=None` and assert SimpleITK parity in the perf-bench tests
instead (cf. `lomb_scargle_interpolate`).

SimpleITK is imported lazily (only the numpy worker runs the sitk baseline), so
the cuda / cupy workers that import this module never need it.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

# --- input generators (verbatim from nitrix's parity tests) ---------------

def phantom(s: int, seed: int = 20260523
            ) -> Tuple[np.ndarray, np.ndarray]:
    '''Concentric-shell tissue phantom with a smooth multiplicative bias --
    `tests/test_bias.py::_phantom`. Returns (observed image, mask).'''
    rng = np.random.default_rng(seed)
    ax = np.linspace(-1, 1, s)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    r = np.sqrt(xx**2 + yy**2 + zz**2)
    tissue = np.where(r < 0.85, 100.0, 0.0)
    tissue = np.where(r < 0.55, 160.0, tissue)
    tissue = np.where(r < 0.25, 210.0, tissue).astype(np.float32)
    mask = (tissue > 0).astype(np.float32)
    bias = np.clip(
        1.0 + 0.45 * np.sin(1.6 * xx + 0.4) + 0.3 * np.cos(1.3 * yy - 0.2)
        + 0.18 * zz, 0.5, 1.8,
    ).astype(np.float32)
    obs = (tissue * bias).astype(np.float32)
    obs = obs + rng.normal(0, 0.8, obs.shape).astype(np.float32) * mask
    return obs.astype(np.float32), mask


def synth_pair(n: int, seed: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    '''Source / reference intensity pair -- `test_histogram_match.py::
    _synth_pair`. Differently-shaped volumes (n^3 vs (n+8)^3).'''
    rng = np.random.default_rng(seed)
    src = rng.uniform(0.0, 1.0, (n, n, n)).astype(np.float32)
    ref = rng.normal(0.5, 0.15, (n + 8, n + 8, n + 8)).astype(np.float32)
    return src, ref


# --- SimpleITK reference kernels (lazy import) -----------------------------

def sitk_n4() -> Callable[[Any, Any], Any]:
    '''ITK `N4BiasFieldCorrectionImageFilter` with the exact parameters
    nitrix's parity test sets (the in-memory array<->Image wrap is included --
    Class-A, no disk I/O). Returns the corrected image.'''

    def run(obs: Any, mask: Any) -> Any:
        import SimpleITK as sitk

        f = sitk.N4BiasFieldCorrectionImageFilter()
        f.SetMaximumNumberOfIterations([50, 50, 50, 50])
        f.SetConvergenceThreshold(1e-3)
        f.SetNumberOfControlPoints([4, 4, 4])
        f.SetNumberOfHistogramBins(200)
        f.SetBiasFieldFullWidthAtHalfMaximum(0.15)
        f.SetWienerFilterNoise(0.01)
        f.SetSplineOrder(3)
        img = sitk.GetImageFromArray(np.asarray(obs))
        msk = sitk.GetImageFromArray(np.asarray(mask).astype(np.uint8))
        return sitk.GetArrayFromImage(f.Execute(img, msk)).astype(np.float32)

    return run


def sitk_histogram_match() -> Callable[[Any, Any], Any]:
    '''ITK `HistogramMatchingImageFilter` (Nyul-Udupa) with the parity test's
    parameters (1024 levels, 7 match points, threshold-at-mean).'''

    def run(src: Any, ref: Any) -> Any:
        import SimpleITK as sitk

        f = sitk.HistogramMatchingImageFilter()
        f.SetNumberOfHistogramLevels(1024)
        f.SetNumberOfMatchPoints(7)
        f.ThresholdAtMeanIntensityOn()
        out = f.Execute(sitk.GetImageFromArray(np.asarray(src)),
                        sitk.GetImageFromArray(np.asarray(ref)))
        return sitk.GetArrayFromImage(out).astype(np.float32)

    return run


# --- morphology / distance floor kernels (ITK = the imaging-standard floor;
#     each verified to match the case's existing fp64 oracle: erode/dilate
#     exact, distance rel_to_tol ~3e-8, median interior-exact). Isotropic, so
#     the ITK (x,y,z) vs numpy (z,y,x) axis flip does not change the result. --

def _box_radius(x: Any, size: int) -> list:
    '''Per-axis radius for an ``size``-wide box (size = 2r+1).'''
    return [(size - 1) // 2] * np.asarray(x).ndim


def sitk_grey_morph(kind: str) -> Callable[[Any, int], Any]:
    '''ITK grayscale erosion / dilation with a flat box structuring element
    (matches ``scipy.ndimage.grey_erosion`` / ``grey_dilation`` exactly).'''

    def run(x: Any, size: int) -> Any:
        import SimpleITK as sitk

        fn = (sitk.GrayscaleErode if kind == 'erode'
              else sitk.GrayscaleDilate)
        out = fn(sitk.GetImageFromArray(np.asarray(x, np.float32)),
                 _box_radius(x, size), sitk.sitkBox)
        return sitk.GetArrayFromImage(out).astype(np.float32)

    return run


def sitk_median(x: Any, size: int) -> Any:
    '''ITK median filter (box radius); matches scipy median in the interior
    (boundary policy differs -- the case has no fp64 oracle, perf only).'''
    import SimpleITK as sitk

    out = sitk.Median(sitk.GetImageFromArray(np.asarray(x, np.float32)),
                      _box_radius(x, size))
    return sitk.GetArrayFromImage(out).astype(np.float32)


def sitk_edt(mask: Any) -> Any:
    '''ITK exact Euclidean distance transform (Danielsson) matching
    ``scipy.ndimage.distance_transform_edt(mask > 0.5)``: distance from each
    foreground voxel to the nearest background voxel. Feed the complement so
    Danielsson's distance-to-nearest-nonzero is the foreground->background EDT,
    non-squared, with unit spacing.'''
    import SimpleITK as sitk

    mb = (np.asarray(mask) > 0.5).astype(np.uint8)
    out = sitk.DanielssonDistanceMap(
        sitk.GetImageFromArray(1 - mb), True, False, True)
    return sitk.GetArrayFromImage(out).astype(np.float32)


# --- parity criteria (verbatim from nitrix's tests) ------------------------

def bias_parity(a: np.ndarray, g: np.ndarray, mask: np.ndarray
                ) -> Tuple[float, float]:
    '''(correlation, scale-invariant relative RMSE) over the mask --
    `tests/test_bias.py::_bias_parity`. N4's field is defined up to a global
    scale, so parity is global, not elementwise.'''
    m = mask > 0
    aa, gg = a[m], g[m]
    corr = float(np.corrcoef(aa, gg)[0, 1])
    s = np.sum(aa * gg) / np.sum(gg * gg)
    rel_rmse = float(np.sqrt(np.mean((aa - s * gg) ** 2)) / gg.mean())
    return corr, rel_rmse
