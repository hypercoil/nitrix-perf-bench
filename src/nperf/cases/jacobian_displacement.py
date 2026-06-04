# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.jacobian_displacement`` vs numpy / cupy.

Per-point Jacobian ``J = I + ∇u`` of the deformation ``φ = id + u`` (central
differences, ``'nearest'`` boundary) -- nitrix (jax) vs a numpy
reimplementation of the same roll-based central diff (CPU floor + fp64 oracle)
+ a CuPy GPU ref. The boundary convention (denominator ``2·spacing`` even at
the one-sided edge cell) matches the interior of ``numpy.gradient`` but not its
boundary, so the exact-convention numpy floor -- verified equal to 0.0 in fp64
-- is the right target, not numpy.gradient. Stencil op (neighbour-coupled),
GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import jacobian_displacement

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_jacobian,
    displacement_input,
    jacobian_sizes,
    np_jacobian,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    U = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(U))
    ref = np_jacobian(U.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(U)
        return (U,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda u: jacobian_displacement(u)),
        'numpy.jacobian': ('numpy', np_jacobian),  # CPU floor
        'cupy.jacobian_displacement': ('cupy', cupy_jacobian()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube edge): input d³·ndim; output d³·ndim² (the per-point Jacobian).
_SIZES = [32, 48, 64]

CASE = Case(
    name='jacobian_displacement',
    op_qualname='nitrix.geometry.jacobian_displacement',
    output_independent=False,  # central diff couples each cell to neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
