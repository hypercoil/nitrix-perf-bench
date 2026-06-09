# -*- coding: utf-8 -*-
"""Tier-2 domain-tool ref: ``nitrix.geometry.resample`` -- the kernel branches.

Image resize (align_corners=True) -- now measured across the **interpolation
kernels** the settled dispatcher exposes via ``method=`` (B18: each dispatch
branch a user hits, measured separately; the dispatcher settled with the
registration R0-R3 merge, 2026-06-09):

- **Linear** / **NearestNeighbour** / **CubicBSpline** have an exact
  ``scipy.ndimage.map_coordinates`` analogue (spline order 1 / 0 / 3; cubic
  with the ``mode='mirror'`` prefilter nitrix forces) -- verified to match to
  fp32 round-off, so scipy is the fp64 oracle + CPU floor and ``cupyx`` the
  GPU bar.  Linear additionally carries the ANTsPy domain reference
  (``resample_image(interp_type=0)``, the ITK engine, matches to 0.0).
- **Lanczos** (windowed-sinc, the high-fidelity ANTs kernel) has **no
  ``map_coordinates`` equivalent**, and nitrix's is the ANTs *algorithm class*,
  not bit-exact ITK parity -- so there is no cross-impl oracle.  It rides as a
  perf-only point (fidelity inconclusive); the value is its dispatch-branch
  *cost* -- the ``2a``-tap windowed-sinc vs the cheaper kernels.

All samples are in-bounds (the grid spans ``[0, in-1]``), so the spline kernels
have no boundary divergence.  Ratio vs ``nitrix-jax``.  ANTs runs in its own
refs env (``NPERF_PYTHON_ANTS``).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.geometry import (
    CubicBSpline,
    Lanczos,
    Linear,
    NearestNeighbour,
    resample,
)

from ._base import BuiltPoint, Case, to_cupy
from ._resample import (
    _KERNEL_ORDER,
    ants_resample,
    cupy_resize,
    resize_coords,
    scipy_resize,
)

_METHOD = {
    'linear': Linear(), 'nearest': NearestNeighbour(),
    'cubic': CubicBSpline(), 'lanczos': Lanczos(3),
}


def _build(param: Dict[str, Any]) -> BuiltPoint:
    in_shape = tuple(param['shape'])
    out_shape = tuple(param['out'])
    kernel = param.get('kernel', 'linear')
    method = _METHOD[kernel]
    rng = np.random.default_rng(param.get('seed', 0))
    img = rng.standard_normal(in_shape + (1,)).astype(np.float32)
    img_j = jax.block_until_ready(jnp.asarray(img))
    coords = resize_coords(in_shape, out_shape)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(img)
        return (img_j,) if framework == 'jax' else (img,)

    baselines: Dict[str, Any] = {
        'nitrix-jax': ('jax', lambda im: resample(im, out_shape,
                                                   method=method)),
    }
    if kernel == 'lanczos':
        # No map_coordinates equivalent + nitrix's is the ANTs algorithm class
        # (not bit-exact ITK parity) -> no cross-impl oracle; perf-only.
        return BuiltPoint(
            baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
            fidelity_note=('Lanczos windowed-sinc: no map_coordinates '
                           'analogue and nitrix is the ANTs algorithm class, '
                           'not bit-exact ITK parity -- perf-only '
                           '(dispatch-branch cost), fidelity inconclusive.'),
            ratio_reference='nitrix-jax',
        )

    order, mode = _KERNEL_ORDER[kernel]
    ref = scipy_resize(coords, order, mode)(img.astype(np.float64))  # oracle
    baselines['scipy.ndimage.map_coordinates'] = (
        'scipy', scipy_resize(coords, order, mode))
    baselines['cupyx.scipy.ndimage.map_coordinates'] = (
        'cupy', cupy_resize(coords, order, mode))  # GPU bar
    if kernel == 'linear':
        baselines['ants.resample_image'] = ('ants', ants_resample(out_shape))
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# Linear across 2-D / 3-D (cost ~ output voxels * taps^ndim); the other kernels
# at the 3-D volume-upsample size, so the dispatch-branch cost is comparable.
_LINEAR = [([128, 128], [256, 256]), ([256, 256], [512, 512]),
           ([64, 64, 64], [128, 128, 128])]
_VOL = ([64, 64, 64], [128, 128, 128])

CASE = Case(
    name='resample',
    op_qualname='nitrix.geometry.resample',
    output_independent=True,  # each output samples a bounded input window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=(
        [{'shape': s, 'out': o, 'kernel': 'linear', 'seed': 0}
         for (s, o) in _LINEAR]
        + [{'shape': _VOL[0], 'out': _VOL[1], 'kernel': k, 'seed': 0}
           for k in ('nearest', 'cubic', 'lanczos')]),
    representative={'shape': [256, 256], 'out': [512, 512],
                    'kernel': 'linear', 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
