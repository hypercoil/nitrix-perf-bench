# -*- coding: utf-8 -*-
"""Shared helpers for the distance-transform family (euclidean + chamfer).

nitrix's ``distance_transform`` has two dispatch branches that must be measured
**independently** (B18 Win 1):

- ``metric='euclidean'`` (default) -- **exact** EDT via a separable per-axis
  min-plus matmul; the reference is ``scipy.ndimage.distance_transform_edt``
  (also exact), so the gate is *tight* (matches to fp32 round-off, ~4e-6 abs on
  realistic distances).  This replaced the historical quasi-Euclidean chamfer
  default whose ~0.4-voxel error needed the old ``atol=1.0`` crutch -- a loose
  gate that would now hide an exact-EDT regression.
- ``metric='chebyshev'`` / ``'city_block'`` -- chamfer DT; **exact for its own
  metric** (chessboard / taxicab), so the reference is
  ``scipy.ndimage.distance_transform_cdt`` and the gate is tight too (matches
  *exactly*; the distances are integers).

Both use a **structured blob mask** (smoothed-noise threshold), not a per-pixel
random mask: a random ``> 0.5`` mask has background everywhere, so every
distance is ~1-2 voxels and the transform never does long-range work (B18's
"degenerate mask / short-circuit").  Smoothing into connected fg/bg regions
gives a realistic interior:boundary ratio and distances that scale with size,
so the number reflects genuine work and the size sweep exposes the
O(n^2)-per-axis matmul crossover honestly.

scipy is a core dep (host); cupy is lazy (refs-cupy worker only) and only
provides an EDT reference -- ``cupyx.scipy.ndimage`` has **no**
``distance_transform_cdt``, so the chamfer case has no on-target GPU reference
(recorded in that case).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.ndimage as spnd


def blob_mask(shape, seed: int = 0, frac: float = 0.5) -> np.ndarray:
    '''Structured blob mask: threshold smoothed noise so foreground/background
    are connected regions with a realistic interior (distances scale with size,
    unlike a per-pixel random mask whose distances are all ~1 voxel).  ``frac``
    is the background quantile (0.5 => ~50% foreground).'''
    rng = np.random.default_rng(seed)
    sigma = max(shape) / 16.0
    field = spnd.gaussian_filter(rng.random(shape).astype(np.float32), sigma)
    thr = float(np.quantile(field, frac))
    return (field > thr).astype(np.float32)


def scipy_edt(m: Any) -> np.ndarray:
    '''Exact Euclidean DT -- the fp64 oracle + CPU floor (distance from each
    foreground voxel to the nearest background voxel).'''
    return spnd.distance_transform_edt(np.asarray(m) > 0.5)


def cupy_edt() -> Callable[[Any], Any]:
    '''GPU exact EDT (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).'''

    def run(m: Any) -> Any:
        from cupyx.scipy import ndimage as cnd

        return cnd.distance_transform_edt(m > 0.5)

    return run


def scipy_cdt(metric: str) -> Callable[[Any], np.ndarray]:
    '''Exact chamfer (chessboard / taxicab) DT -- the fp64 oracle + CPU floor
    for the chamfer engine.  Returns float32 (cdt yields integer distances).'''

    def run(m: Any) -> np.ndarray:
        out = spnd.distance_transform_cdt(np.asarray(m) > 0.5, metric=metric)
        return out.astype(np.float32)

    return run
