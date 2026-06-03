# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.displacement_from_reference_points``.

The point-cloud centre-of-mass displacement from a per-region reference --
``center_of_mass_points(W, X) - reference`` -- nitrix (jax) vs the numpy
weighted mean minus the same reference (CPU floor + fp64 oracle) + a CuPy GPU
ref. The reference is the global coordinate centroid (a fixed parameter, baked
outside the timed region). Matmul + reduction, GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import displacement_from_reference_points

from ._base import BuiltPoint, Case, to_cupy
from ._geometry import (
    cupy_displacement_points,
    np_com_points,
    np_displacement_points,
    points_input,
)

_REGIONS = 64
_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = param['p']
    W, X = points_input(_REGIONS, p, _NDIM, param.get('seed', 0))
    reference = X.astype('float64').mean(0)  # (ndim,), broadcasts over regions
    jW = jax.block_until_ready(jnp.asarray(W))
    jX = jax.block_until_ready(jnp.asarray(X))
    jref = jax.block_until_ready(jnp.asarray(reference))
    ref = np_com_points(W.astype('float64'), X.astype('float64')) - reference

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W, X)
        return (W, X) if framework == 'numpy' else (jW, jX)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda w, x: displacement_from_reference_points(w, jref, x)),
        'numpy.weighted_mean': (
            'numpy', np_displacement_points(reference)),  # CPU floor
        'cupy.displacement_from_reference_points': (
            'cupy', cupy_displacement_points(reference)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ regions·points·ndim (the W @ X matmul); regions = _REGIONS.
_SIZES = [1024, 4096, 16384]

CASE = Case(
    name='displacement_from_reference_points',
    op_qualname='nitrix.geometry.displacement_from_reference_points',
    output_independent=False,  # each region's centroid reduces over all points
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'p': p, 'seed': 0} for p in _SIZES],
    representative={'p': 4096, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
