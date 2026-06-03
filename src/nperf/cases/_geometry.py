# -*- coding: utf-8 -*-
"""Shared helpers for the geometry centre-of-mass cluster.

``nitrix.geometry.{center_of_mass_grid, center_of_mass_points,
displacement_from_reference_grid, displacement_from_reference_points}`` are all
weighted-centroid reductions:

- **grid**: centre of mass of a dense weight volume, treating each cell's
  index as its coordinate -- ``cm[d] = Σ_i i·w / Σ w``. This is *exactly*
  ``scipy.ndimage.center_of_mass`` (verified to ~1e-14 in fp64), the canonical
  medical-imaging / array reference (already a core dep).
- **points**: weighted mean of a point cloud per region -- ``cm = (W @ X) /
  W.sum(-1)``; no grid/ndimage equivalent, so a numpy weighted-mean floor.
- **displacement_***: the centre of mass minus a reference coordinate (the
  registration regulariser pattern) -- ``com - reference``.

All are pure reduction/matmul/broadcast -> GPU-pure (no solver). The cupy refs
reimplement the same index-weighted reduction (a fair kernel-vs-kernel GPU bar,
version-independent of ``cupyx.scipy.ndimage``). scipy.ndimage is a core dep;
cupy is lazy (its worker only).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np


def volume_input(shape: Sequence[int], seed: int = 0) -> np.ndarray:
    '''A dense non-negative weight volume (the centre-of-mass input).'''
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, tuple(shape)).astype(np.float32)


def points_input(
    n_regions: int, n_points: int, ndim: int = 3, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    '''A point-cloud assignment: weights ``(n_regions, n_points)`` (non-neg) +
    coordinates ``(n_points, ndim)``.'''
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, (n_regions, n_points)).astype(np.float32)
    x = rng.standard_normal((n_points, ndim)).astype(np.float32)
    return w, x


def _com_grid(w: Any, xp: Any) -> Any:
    '''Index-weighted centroid over all axes (nitrix's algorithm), in ``xp``;
    returns an ``(ndim,)`` vector matching ``scipy.ndimage.center_of_mass``.'''
    total = w.sum()
    out = []
    for ax in range(w.ndim):
        shape = [1] * w.ndim
        shape[ax] = w.shape[ax]
        coor = xp.arange(w.shape[ax], dtype=w.dtype).reshape(shape)
        out.append((coor * w).sum() / total)
    return xp.stack(out)


def np_com_grid(w: Any) -> np.ndarray:
    '''numpy index-weighted centroid (fp64 oracle).'''
    return _com_grid(np.asarray(w), np)


def np_com_points(w: Any, x: Any) -> np.ndarray:
    '''numpy weighted mean per region ``(W @ X) / W.sum(-1)`` (floor/oracle).'''
    w = np.asarray(w)
    x = np.asarray(x)
    return (w @ x) / w.sum(-1, keepdims=True)


def scipy_com_grid() -> Callable[[Any], Any]:
    '''``scipy.ndimage.center_of_mass`` -- the canonical array/imaging
    reference (CPU floor); already a core dep.'''
    import scipy.ndimage as ndi

    def run(w: Any) -> Any:
        return np.array(ndi.center_of_mass(np.asarray(w)))

    return run


def scipy_displacement_grid(reference: np.ndarray) -> Callable[[Any], Any]:
    '''``scipy.ndimage.center_of_mass`` minus the reference -- the canonical
    floor for ``displacement_from_reference_grid``.'''
    import scipy.ndimage as ndi

    def run(w: Any) -> Any:
        return np.array(ndi.center_of_mass(np.asarray(w))) - reference

    return run


def np_displacement_points(reference: np.ndarray) -> Callable[[Any, Any], Any]:
    '''numpy weighted mean minus the per-region reference (floor/oracle).'''

    def run(w: Any, x: Any) -> Any:
        return np_com_points(w, x) - reference

    return run


def cupy_com_grid() -> Callable[[Any], Any]:
    '''GPU index-weighted centroid (same algorithm); cupy lazy.'''

    def run(w: Any) -> Any:
        import cupy as cp

        return _com_grid(w, cp)

    return run


def cupy_com_points() -> Callable[[Any, Any], Any]:
    '''GPU weighted mean per region; cupy lazy.'''

    def run(w: Any, x: Any) -> Any:
        return (w @ x) / w.sum(-1, keepdims=True)

    return run


def cupy_displacement_grid(reference: np.ndarray) -> Callable[[Any], Any]:
    '''GPU centre-of-mass minus reference; cupy lazy.'''

    def run(w: Any) -> Any:
        import cupy as cp

        return _com_grid(w, cp) - cp.asarray(reference)

    return run


def cupy_displacement_points(
    reference: np.ndarray,
) -> Callable[[Any, Any], Any]:
    '''GPU weighted mean minus reference; cupy lazy.'''

    def run(w: Any, x: Any) -> Any:
        import cupy as cp

        return (w @ x) / w.sum(-1, keepdims=True) - cp.asarray(reference)

    return run
