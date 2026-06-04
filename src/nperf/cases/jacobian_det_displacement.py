# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.jacobian_det_displacement`` vs numpy.

Per-point ``det(I + ∇u)`` -- the folding-detection QA scalar (``det > 0`` is
orientation-preserving) -- nitrix (jax, closed-form Sarrus for d=3) vs a numpy
reimplementation of the same central-diff + closed-form determinant (CPU floor
+ fp64 oracle) + a CuPy GPU ref. Same ``'nearest'`` boundary convention as
``jacobian_displacement`` (verified equal to 0.0 in fp64). Stencil op,
GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import jacobian_det_displacement

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_jac_det,
    displacement_input,
    jacobian_sizes,
    np_jac_det,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    U = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(U))
    ref = np_jac_det(U.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(U)
        return (U,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda u: jacobian_det_displacement(u)),
        'numpy.jacobian_det': ('numpy', np_jac_det),  # CPU floor
        'cupy.jacobian_det_displacement': (
            'cupy', cupy_jac_det()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube edge): input d³·ndim; output d³ (the per-point determinant).
_SIZES = [32, 48, 64]

CASE = Case(
    name='jacobian_det_displacement',
    op_qualname='nitrix.geometry.jacobian_det_displacement',
    output_independent=False,  # central diff couples each cell to neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
