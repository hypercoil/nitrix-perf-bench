# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.modularity_matrix`` vs networkx / cupy.

The Newman modularity matrix ``B = A - k kᵀ / 2m`` -- nitrix (jax) vs
**networkx**'s ``modularity_matrix(G, weight='weight')`` (the canonical graph
library; the *weighted* degree/null term -- the default ``weight=None`` uses
the binary adjacency and does NOT match; verified exact in fp64) + a CuPy GPU
reference, scored against an fp64 numpy oracle.

The networkx floor builds the Graph from the array (the honest end-to-end
graph-object cost), so the headline is the GPU comparison + nitrix's
array-native advantage. Pure outer-product/broadcast, so GPU-pure. Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.graph import modularity_matrix

from ._base import BuiltPoint, Case, to_cupy
from ._graph import cupy_modularity, graph_input, np_modularity, nx_modularity


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    A = graph_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = np_modularity(A.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': (
            'jax', lambda a: modularity_matrix(a, gamma=1.0, normalise=False)),
        'networkx.modularity_matrix': ('networkx', nx_modularity()),  # floor
        'cupy.modularity_matrix': ('cupy', cupy_modularity()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (nodes): cost ~ n² (the dense outer product); sparse-structured adjacency.
_SIZES = [128, 256, 512]

CASE = Case(
    name='modularity_matrix',
    op_qualname='nitrix.graph.modularity_matrix',
    output_independent=False,  # the rank-1 null term couples every entry
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
