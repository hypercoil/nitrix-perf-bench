# -*- coding: utf-8 -*-
"""Geometry stencil: ``nitrix.geometry.spatial_gradient`` vs numpy / cupy.

The spatial gradient of a scalar field along each axis (``(*spatial,) ->
(*spatial, ndim)``) -- the default ``scheme='central'`` / ``mode='nearest'``
branch -- the per-axis central difference the diffeomorphic-Demons force
``½(∇F + ∇(M∘φ))`` is built from.  A single roll-based central-diff pass.

Warranted comparison: the numpy reimplementation is that exact central diff
(``'nearest'`` boundary, denominator ``2·spacing`` -- verified ~1.2e-7), the
fp64 oracle + CPU floor, **not** ``numpy.gradient`` (which differs at the
boundary); cupy is the on-target GPU bar.  Stencil (neighbour-coupled),
memory-bandwidth-bound, GPU-pure.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import spatial_gradient

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_spatial_gradient,
    jacobian_sizes,
    np_spatial_gradient,
    scalar_field_input,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    F = scalar_field_input(jacobian_sizes(d), param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(F))
    ref = np_spatial_gradient(F.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(F)
        return (F,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda f: spatial_gradient(f)),
        'numpy.spatial_gradient': ('numpy', np_spatial_gradient),
        'cupy.spatial_gradient': ('cupy', cupy_spatial_gradient()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='spatial_gradient',
    op_qualname='nitrix.geometry.spatial_gradient',
    output_independent=False,  # central diff couples each cell to neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the voxel count N: one roll-based central-diff pass per '
        'spatial axis (ndim passes), each a shift + subtract. Pure '
        'stencil, memory-bandwidth-bound and GPU-pure (no solver); HBM ~ N '
        '(the ndim-component output). The size tier varies the volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
