# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.spherical_conv`` vs numpy / cupy.

Convolve per-point data on a 2-sphere with an isotropic Gaussian kernel: build
a per-point geodesic kNN, weight neighbours by a Gaussian over geodesic
distance, row-normalise, reduce -- nitrix (jax, ``neighbourhood=k`` on-the-fly
path) vs a numpy reimplementation of the same kNN-Gaussian reduction (CPU floor
+ fp64 oracle) + a CuPy GPU ref. A nitrix-specific surface-smoothing op (no
external library), verified to ~5e-7 in fp64. The O(n²) geodesic + top-k
dominates; GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import spherical_conv

from ._base import BuiltPoint, Case, to_cupy
from ._sphere import (
    conv_data_input,
    cupy_spherical_conv,
    np_spherical_conv,
    xyz_input,
)

_C = 16       # feature channels
_K = 16       # neighbourhood size
_SIGMA = 0.2  # Gaussian sigma (radians, r=1)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    seed = param.get('seed', 0)
    DATA = conv_data_input(n, _C, seed)
    COOR = xyz_input(n, seed + 1)  # unit-sphere points (n, 3)
    jD = jax.block_until_ready(jnp.asarray(DATA))
    jC = jax.block_until_ready(jnp.asarray(COOR))
    ref = np_spherical_conv(_SIGMA, _K)(
        DATA.astype('float64'), COOR.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(DATA, COOR)
        return (DATA, COOR) if framework == 'numpy' else (jD, jC)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda data, coor: spherical_conv(
                data, coor, sigma=_SIGMA, neighbourhood=_K)),
        'numpy.spherical_conv': (
            'numpy', np_spherical_conv(_SIGMA, _K)),  # CPU floor
        'cupy.spherical_conv': (
            'cupy', cupy_spherical_conv(_SIGMA, _K)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ n² (the all-pairs geodesic + top-k); c = _C, k = _K.
_SIZES = [256, 512, 1024]

CASE = Case(
    name='spherical_conv',
    op_qualname='nitrix.geometry.spherical_conv',
    output_independent=False,  # each point reduces over its k neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 512, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
