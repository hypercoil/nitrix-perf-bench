# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.integrate_velocity_field``.

vs scipy / cupy. The diffeomorphic exponential map by scaling-and-squaring
(voxelmorph ``VecInt``): ``φ = v/2ⁿ`` then ``φ ← φ + φ∘(id+φ)`` for ``n_steps``
doublings, each composition a linear-interpolation warp -- nitrix (jax) vs a
numpy reimplementation whose composition core is
**``scipy.ndimage.map_coordinates``** (order=1, mode='nearest'; the genuine
domain-tool interpolation, a core dep) +
a CuPy GPU ref using ``cupyx.scipy.ndimage.map_coordinates``. The full
scaling-squaring loop reproduces nitrix's flow to ~5e-16 in fp64. The op also
exercises ``identity_grid`` internally (the id + φ deformation each step).
GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import integrate_velocity_field

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_integrate,
    displacement_input,
    jacobian_sizes,
    np_integrate,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    # a stationary velocity field (reuse the small-vector-field generator)
    V = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0),
                           scale=0.2)
    jx = jax.block_until_ready(jnp.asarray(V))
    ref = np_integrate(V.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(V)
        return (V,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda v: integrate_velocity_field(v)),
        'scipy.ndimage.map_coordinates': ('scipy', np_integrate),  # CPU floor
        'cupy.integrate_velocity_field': (
            'cupy', cupy_integrate()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube edge): n_steps=7 doublings, each an ndim-channel interp warp; kept
# smaller than the jacobian cases since the integration is iterative.
_SIZES = [16, 24, 32]

CASE = Case(
    name='integrate_velocity_field',
    op_qualname='nitrix.geometry.integrate_velocity_field',
    output_independent=False,  # the warp composition couples neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 24, 'seed': 0},
    build=_build,
    rtol=2e-3,
    atol=2e-4,
)
