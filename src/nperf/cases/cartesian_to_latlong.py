# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.cartesian_to_latlong`` vs numpy / cupy.

Cartesian ``(x, y, z)`` -> ``(latitude, longitude)`` in radians (the inverse of
``latlong_to_cartesian``) -- nitrix (jax) vs the numpy closed form (CPU floor +
fp64 oracle) + a CuPy GPU ref. Pure elementwise ``arctan2`` with an unambiguous
formula, so the closed form IS the reference (verified: the round-trip recovers
the input angles to ~1e-16 in fp64). GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import cartesian_to_latlong

from ._base import BuiltPoint, Case, to_cupy
from ._sphere import (
    cupy_cartesian_to_latlong,
    np_cartesian_to_latlong,
    xyz_input,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    XYZ = xyz_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(XYZ))
    ref = np_cartesian_to_latlong(XYZ.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(XYZ)
        return (XYZ,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda xyz: cartesian_to_latlong(xyz)),
        'numpy.cartesian_to_latlong': (
            'numpy', np_cartesian_to_latlong),  # CPU floor
        'cupy.cartesian_to_latlong': (
            'cupy', cupy_cartesian_to_latlong()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ n (elementwise arctan2); (n, 3) -> (n, 2).
_SIZES = [4096, 16384, 65536]

CASE = Case(
    name='cartesian_to_latlong',
    op_qualname='nitrix.geometry.cartesian_to_latlong',
    output_independent=True,  # each point's conversion is independent
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 16384, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
