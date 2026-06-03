# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.center_of_mass_points`` vs numpy / cupy.

Weighted centre of mass of a point cloud per region -- ``(W @ X) /
W.sum(-1)`` -- nitrix (jax) vs the textbook numpy weighted mean (CPU floor +
fp64 oracle) + a CuPy GPU ref. No grid/ndimage equivalent (this is the
unstructured point-cloud form), so the reference is the array
reimplementation. Matmul + reduction, so GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import center_of_mass_points

from ._base import BuiltPoint, Case, to_cupy
from ._geometry import cupy_com_points, np_com_points, points_input

_REGIONS = 64
_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = param['p']
    W, X = points_input(_REGIONS, p, _NDIM, param.get('seed', 0))
    jW = jax.block_until_ready(jnp.asarray(W))
    jX = jax.block_until_ready(jnp.asarray(X))
    ref = np_com_points(W.astype('float64'), X.astype('float64'))  # oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W, X)
        return (W, X) if framework == 'numpy' else (jW, jX)

    baselines = {
        'nitrix-jax': ('jax', lambda w, x: center_of_mass_points(w, x)),
        'numpy.weighted_mean': ('numpy', np_com_points),  # CPU floor
        'cupy.center_of_mass_points': ('cupy', cupy_com_points()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ regions·points·ndim (the W @ X matmul); regions = _REGIONS.
_SIZES = [1024, 4096, 16384]

CASE = Case(
    name='center_of_mass_points',
    op_qualname='nitrix.geometry.center_of_mass_points',
    output_independent=False,  # each region's centroid reduces over all points
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'p': p, 'seed': 0} for p in _SIZES],
    representative={'p': 4096, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
