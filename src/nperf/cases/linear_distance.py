# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.linear_distance`` vs sklearn / cupy.

The pairwise **squared** L2 distance ``‖x - y‖²`` of a feature matrix: nitrix
(jax) vs ``sklearn.metrics.pairwise.euclidean_distances(squared=True)`` (the
canonical CPU floor) + a CuPy GPU reference (Gram identity, clipped; see
``cases/_kernels.py``), scored against an fp64 oracle (scipy ``cdist`` --
direct, no cancellation).  Matmul/broadcast based, so GPU-pure (no cuSolver): a
clean apples-to-apples GPU bar.  Ratio vs the sklearn distance.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.linalg import linear_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

from ._base import BuiltPoint, Case, to_cupy
from ._kernels import cupy_kernel, kernel_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d = param['n'], param['d']
    X = kernel_input(n, d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))

    Xd = X.astype(np.float64)
    ref = cdist(Xd, Xd, 'sqeuclidean')  # fp64 oracle (direct)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: linear_distance(x)),
        'sklearn.euclidean_distances': (
            'sklearn', lambda x: euclidean_distances(x, squared=True)),
        'cupy.linear_distance': ('cupy', cupy_kernel('distance')),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features): cost ~ n² · d (matmul-bound); output is n x n.
_SHAPES = [(512, 64), (2048, 64), (4096, 64)]

CASE = Case(
    name='linear_distance',
    op_qualname='nitrix.linalg.linear_distance',
    # D[i, j] = ‖xᵢ - xⱼ‖² couples rows i and j; not element-wise independent,
    # but the fp64 oracle is computed in full (cheap) -- documentary (§C).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'seed': 0} for (n, d) in _SHAPES],
    representative={'n': 2048, 'd': 64, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
