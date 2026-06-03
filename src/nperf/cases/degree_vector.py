# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.degree_vector`` vs numpy / cupy.

Per-node weighted degree (row sum of the adjacency) -- nitrix (jax) vs the
textbook numpy row-sum (CPU floor + fp64 oracle) + a CuPy GPU reference. A
trivial reduction, so memory-bound and GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.graph import degree_vector

from ._base import BuiltPoint, Case, to_cupy
from ._graph import cupy_degree, graph_input, np_degree


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    A = graph_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = np_degree(A.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: degree_vector(a)),
        'numpy.degree': ('numpy', np_degree),  # CPU floor
        'cupy.degree': ('cupy', cupy_degree()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [128, 256, 512]

CASE = Case(
    name='degree_vector',
    op_qualname='nitrix.graph.degree_vector',
    output_independent=True,  # each output is one node's independent row sum
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
