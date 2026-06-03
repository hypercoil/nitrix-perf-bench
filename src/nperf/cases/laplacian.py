# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.laplacian`` vs scipy.sparse.csgraph / cupy.

The combinatorial graph Laplacian ``L = D - A`` -- nitrix (jax) vs
``scipy.sparse.csgraph.laplacian(normed=False)`` (the array-based scientific
standard; verified exact in fp64) + a CuPy GPU reference, scored against an
fp64 oracle. Pure broadcast/reduction, so GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.graph import laplacian

from ._base import BuiltPoint, Case, to_cupy
from ._graph import cupy_laplacian, graph_input, scipy_laplacian


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    A = graph_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = scipy_laplacian()(A.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': (
            'jax', lambda a: laplacian(a, normalisation='combinatorial')),
        'scipy.csgraph.laplacian': ('scipy', scipy_laplacian()),  # CPU floor
        'cupy.laplacian': ('cupy', cupy_laplacian()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (nodes): cost ~ n² (the dense D - A); sparse-structured adjacency.
_SIZES = [128, 256, 512]

CASE = Case(
    name='laplacian',
    op_qualname='nitrix.graph.laplacian',
    output_independent=False,  # the diagonal couples each row's degree
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
