# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.girvan_newman_null`` vs numpy / cupy.

The rank-1 Girvan-Newman null model ``k kᵀ / 2m`` (the expected-edges term in
modularity) -- nitrix (jax) vs the textbook numpy outer-product (CPU floor +
fp64 oracle) + a CuPy GPU reference. An outer product, so GPU-pure. Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.graph import girvan_newman_null

from ._base import BuiltPoint, Case, to_cupy
from ._graph import cupy_gn_null, graph_input, np_gn_null


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    A = graph_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = np_gn_null(A.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: girvan_newman_null(a)),
        'numpy.gn_null': ('numpy', np_gn_null),  # CPU floor
        'cupy.gn_null': ('cupy', cupy_gn_null()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [128, 256, 512]

CASE = Case(
    name='girvan_newman_null',
    op_qualname='nitrix.graph.girvan_newman_null',
    output_independent=False,  # the rank-1 outer product couples every entry
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
