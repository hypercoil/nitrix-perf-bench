# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.compactness_penalty`` vs numpy / cupy.

How dispersed is each region's weight from its own centre of mass --
``mean_p w·‖cm - x_p‖₂`` (Euclidean, ``radius=None``) -- the registration
regulariser that pulls region weights toward compact supports. nitrix (jax) vs
the numpy reimplementation (CPU floor + fp64 oracle) + a CuPy GPU ref. A
community/registration-specific reduction (no external library), built on the
``center_of_mass_points`` Gram. Matmul + reduction, GPU-pure. Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import compactness_penalty

from ._base import BuiltPoint, Case, to_cupy
from ._geometry import cupy_compactness, np_compactness, points_input

_REGIONS = 64
_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = param['p']
    W, X = points_input(_REGIONS, p, _NDIM, param.get('seed', 0))
    jW = jax.block_until_ready(jnp.asarray(W))
    jX = jax.block_until_ready(jnp.asarray(X))
    ref = np_compactness(W.astype('float64'), X.astype('float64'))  # oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W, X)
        return (W, X) if framework == 'numpy' else (jW, jX)

    baselines = {
        'nitrix-jax': ('jax', lambda w, x: compactness_penalty(w, x)),
        'numpy.compactness': ('numpy', np_compactness),  # CPU floor
        'cupy.compactness_penalty': ('cupy', cupy_compactness()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ regions·points·ndim (the cm Gram + per-point distance);
# regions = _REGIONS.
_SIZES = [1024, 4096, 16384]

CASE = Case(
    name='compactness_penalty',
    op_qualname='nitrix.geometry.compactness_penalty',
    output_independent=False,  # each region reduces over all its points
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'p': p, 'seed': 0} for p in _SIZES],
    representative={'p': 4096, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
