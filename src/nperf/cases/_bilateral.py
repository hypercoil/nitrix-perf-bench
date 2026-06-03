# -*- coding: utf-8 -*-
"""Shared helper: bilateral_gaussian configured as a grid image bilateral.

nitrix's ``bilateral_gaussian`` is a general *point-cloud* bilateral (per-point
values smoothed over a feature-space metric on a bounded neighbourhood). ITK /
SimpleITK's ``BilateralImageFilter`` is the *image* special case: a regular
grid, a box spatial window, a domain Gaussian (sigma_d, pixels) x a range
Gaussian (sigma_r, intensity). We configure the nitrix op to reproduce it
exactly:

- **window** = box of radius ``r = ceil(domainMu * sigma_d)`` (ITK's
  ``domainMu = 2.5``, verified empirically) via ``regular_grid_stencil``;
- **features** = ``[row, col, intensity]`` per pixel;
- **metric** = ``DiagonalMetric([sigma_d, sigma_d, sigma_r])`` -> the weight
  ``exp(-1/2 * sum((f_i-f_j)/sigma)**2)`` == ITK's domain x range Gaussian.

**Parity (verified).** The interior matches ``sitk.Bilateral`` to ~1e-4 -- the
bounded window, both Gaussians, and the sum-of-weights normalisation all match.
Only the r-pixel **boundary** diverges (ITK's edge handling differs from
nitrix's ``replicate``/``reflect`` stencil boundary), so the case carries no
fp64 oracle and asserts *interior* parity instead (cf. ``median_filter``).

SimpleITK is imported lazily (only the numpy worker runs the sitk baseline).
"""
from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import numpy as np

_DOMAIN_MU = 2.5  # ITK BilateralImageFilter domain truncation (window radius)
_RANGE_SAMPLES = 100000  # ITK range-Gaussian LUT size; high -> ~exact exp


def bilateral_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    '''A (h x w) intensity image in [0, 1] -- enough intensity structure that
    the range Gaussian actually gates the spatial average.'''
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (h, w)).astype(np.float32)


def grid_bilateral_setup(img: np.ndarray, sigma_d: float, sigma_r: float
                         ) -> Tuple[Any, Any, Any, Any, int]:
    '''Return (values, features, ell, metric, radius) configuring
    ``bilateral_gaussian`` as the ITK-matching grid image bilateral.'''
    import jax.numpy as jnp
    from nitrix.smoothing.metric import DiagonalMetric
    from nitrix.sparse import regular_grid_stencil

    h, w = img.shape
    r = math.ceil(_DOMAIN_MU * sigma_d)
    offsets = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)]
    ell = regular_grid_stencil(
        (h, w), offsets, np.ones(len(offsets), np.float32),
        boundary='replicate')
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    feats = np.stack([rr.ravel(), cc.ravel(), img.ravel()], 1).astype(
        np.float32)
    vals = img.ravel()[:, None].astype(np.float32)
    metric = DiagonalMetric(
        jnp.asarray([sigma_d, sigma_d, sigma_r], jnp.float32))
    return vals, feats, ell, metric, r


def sitk_bilateral(sigma_d: float, sigma_r: float) -> Callable[[Any], Any]:
    '''ITK ``BilateralImageFilter`` (the image-bilateral reference); high
    range-sample count so its sampled range Gaussian ~= the exact exp.'''

    def run(img: Any) -> Any:
        import SimpleITK as sitk

        out = sitk.Bilateral(
            sitk.GetImageFromArray(np.asarray(img, np.float32)),
            sigma_d, sigma_r, _RANGE_SAMPLES)
        return sitk.GetArrayFromImage(out).astype(np.float32)

    return run
