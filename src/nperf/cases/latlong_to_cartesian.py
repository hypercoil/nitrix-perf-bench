# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.latlong_to_cartesian`` vs numpy / cupy.

Latitude/longitude (radians) -> Cartesian ``(x, y, z)`` on a sphere -- nitrix
(jax) vs the numpy closed form (CPU floor + fp64 oracle) + a CuPy GPU ref. Pure
elementwise trig with an unambiguous formula, so the closed form IS the
reference (no external library is more canonical; verified equal to 0.0 in
fp64). GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import latlong_to_cartesian

from ._base import BuiltPoint, Case, to_cupy
from ._sphere import (
    cupy_latlong_to_cartesian,
    latlong_input,
    np_latlong_to_cartesian,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    LL = latlong_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(LL))
    ref = np_latlong_to_cartesian(LL.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(LL)
        return (LL,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda ll: latlong_to_cartesian(ll)),
        'numpy.latlong_to_cartesian': (
            'numpy', np_latlong_to_cartesian),  # CPU floor
        'cupy.latlong_to_cartesian': (
            'cupy', cupy_latlong_to_cartesian()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ n (elementwise trig); (n, 2) -> (n, 3).
_SIZES = [4096, 16384, 65536]

CASE = Case(
    name='latlong_to_cartesian',
    op_qualname='nitrix.geometry.latlong_to_cartesian',
    output_independent=True,  # each point's conversion is independent
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 16384, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
