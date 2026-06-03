# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.relaxed_modularity`` vs networkx / cupy.

The (differentiable) relaxed modularity quality score ``Q = sum_ij B_ij
(C Cᵀ)_ij / 2`` -- a scalar -- for a graph ``A`` and a community-assignment
``C``. We benchmark the **hard one-hot** partition + ``exclude_diag=False``:
the canonical-comparable mode, where the score equals
``networkx.community.modularity(G, weight='weight') / 2`` (verified exact in
fp64). The ``/2`` is a real convention bridge -- nitrix corrects the
undirected double-count *twice* (the ``1/2m`` prefactor + an explicit
``Q/2``), so its score is the literature Newman modularity halved (the op
docstring's "reduces to the standard Newman modularity" is off by this factor;
filed low-priority with nitrix). The default ``exclude_diag=True`` additionally
drops the within-community diagonal term -- so it has no clean external
reference and is not the mode benchmarked here.

networkx is the canonical CPU reference (the Newman quality score everyone
benchmarks against); the GPU comparison is nitrix-jax vs a dense CuPy
reimplementation. The op consumes the rank-1 null structure (matmul + reduction
to a scalar), so GPU-pure (no solver). Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import relaxed_modularity

from ._base import BuiltPoint, Case, to_cupy
from ._graph import (
    cupy_relaxed_modularity,
    graph_input,
    np_relaxed_modularity,
    nx_relaxed_modularity,
    partition_input,
)

_K = 8  # number of (planted-at-random) communities in the hard partition


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    seed = param.get('seed', 0)
    A = graph_input(n, seed)
    C = partition_input(n, _K, seed)
    jA = jax.block_until_ready(jnp.asarray(A))
    jC = jax.block_until_ready(jnp.asarray(C))
    ref = np_relaxed_modularity(A.astype('float64'),
                                C.astype('float64'))  # fp64 oracle (scalar)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A, C)
        return (A, C) if framework == 'numpy' else (jA, jC)

    baselines = {
        'nitrix-jax': (
            'jax', lambda a, c: relaxed_modularity(a, c, exclude_diag=False)),
        'networkx.modularity': ('networkx', nx_relaxed_modularity()),  # floor
        'cupy.relaxed_modularity': (
            'cupy', cupy_relaxed_modularity()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (nodes): cost ~ n² (dense B · C Cᵀ); sparse-structured adjacency, k = _K.
_SIZES = [128, 256, 512]

CASE = Case(
    name='relaxed_modularity',
    op_qualname='nitrix.graph.relaxed_modularity',
    output_independent=False,  # a single scalar reduction over the whole graph
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
