# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.spherical_geodesic_distance``.

vs sklearn / cupy. All-pairs great-circle distance between Cartesian points on
a sphere, via the robust ``r·atan2(|X×Y|, X·Y)`` formula -- nitrix (jax) vs
**``sklearn.metrics.pairwise.haversine_distances``** on the points' lat/long
(the canonical domain-tool great-circle distance; for the unit sphere the
angular distance equals the geodesic, verified ~2e-15 in fp64) + a CuPy GPU ref
reimplementing the same atan2 formula, scored against an fp64 oracle. All-pairs
broadcast, so GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import spherical_geodesic_distance

from ._base import BuiltPoint, Case, to_cupy
from ._sphere import cupy_geodesic, np_geodesic, sklearn_haversine, xyz_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    X = xyz_input(n, param.get('seed', 0))  # unit-sphere points (n, 3)
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_geodesic(X.astype('float64'))  # fp64 oracle (self-pairwise)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: spherical_geodesic_distance(x)),
        'sklearn.haversine': ('sklearn', sklearn_haversine()),  # CPU floor
        'cupy.spherical_geodesic_distance': (
            'cupy', cupy_geodesic()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (points): cost ~ n² (the all-pairs distance matrix).
_SIZES = [256, 512, 1024]

CASE = Case(
    name='spherical_geodesic_distance',
    op_qualname='nitrix.geometry.spherical_geodesic_distance',
    output_independent=True,  # each (i, j) distance is independent
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 512, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
