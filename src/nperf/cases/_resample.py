# -*- coding: utf-8 -*-
"""Shared helpers for the resample (image-resize) case.

``nitrix.geometry.resample`` resizes a channel-last image to ``target_shape``
by **linear interpolation with align_corners=True** (output ``i`` samples input
``i*(in-1)/(out-1)``). That convention is shared exactly by:

- **ANTsPy** ``resample_image(..., interp_type=0)`` -- the canonical
  medical-imaging reference (the ITK engine; verified to match nitrix to 0.0),
  the *genuine* ANTsPy in-memory op (unlike file-coupled ``apply_transforms``);
- ``scipy.ndimage.map_coordinates`` (CPU floor) + ``cupyx`` (GPU ref) on the
  same align_corners sample grid.

All samples are in-bounds (the grid spans ``[0, in-1]``), so -- unlike the warp
/ median / bilateral cases -- there is **no boundary divergence**: a clean fp64
oracle, and every baseline (ANTs included) matches it.

ants / cupy / scipy are imported lazily (their own workers only).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def resize_coords(in_shape: Sequence[int], out_shape: Sequence[int]
                  ) -> np.ndarray:
    '''align_corners=True sample coordinates: output index ``i`` along each
    axis samples the input at ``i*(in-1)/(out-1)``. Shape ``(ndim, *out)``.'''
    axes = [np.linspace(0.0, s_in - 1, s_out, dtype=np.float64)
            for s_in, s_out in zip(in_shape, out_shape)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=0)


# Each interpolation kernel maps to a ``map_coordinates`` spline ``order`` and
# the prefilter boundary ``mode`` that matches nitrix's kernel (verified):
#   Linear -> order 1, NearestNeighbour -> order 0, CubicBSpline -> order 3
#   with ``mode='mirror'`` (nitrix forces the mirror spline prefilter).
#   Lanczos has no ``map_coordinates`` equivalent (and nitrix's is the ANTs
#   algorithm class, not bit-exact ITK parity), so it has no cross-impl oracle.
_KERNEL_ORDER = {'linear': (1, 'nearest'), 'nearest': (0, 'nearest'),
                 'cubic': (3, 'mirror')}


def scipy_resize(coords: np.ndarray, order: int = 1,
                 mode: str = 'nearest') -> Callable[[Any], Any]:
    '''Resize via ``scipy.ndimage.map_coordinates`` on the align_corners grid
    (CPU floor + fp64 oracle for the spline kernels); channel-last.'''
    import scipy.ndimage as spnd

    def run(img: Any) -> Any:
        out = spnd.map_coordinates(np.asarray(img)[..., 0], coords,
                                   order=order, mode=mode)
        return out[..., None]

    return run


def cupy_resize(coords: np.ndarray, order: int = 1,
                mode: str = 'nearest') -> Callable[[Any], Any]:
    '''GPU twin of ``scipy_resize`` (cupyx.scipy.ndimage); cupy lazy.'''

    def run(img: Any) -> Any:
        import cupy as cp
        from cupyx.scipy import ndimage as cnd

        out = cnd.map_coordinates(img[..., 0], cp.asarray(coords),
                                  order=order, mode=mode)
        return out[..., None]

    return run


def ants_resample(out_shape: Sequence[int]) -> Callable[[Any], Any]:
    '''ANTsPy ``resample_image`` (linear, interp_type=0) -- the domain-tool
    reference; ants lazy (only the ants worker imports it).'''
    target = tuple(int(s) for s in out_shape)

    def run(img: Any) -> Any:
        import ants

        a = ants.from_numpy(np.asarray(img, np.float32)[..., 0])
        out = ants.resample_image(a, target, True, 0).numpy()
        return out[..., None].astype(np.float32)

    return run
