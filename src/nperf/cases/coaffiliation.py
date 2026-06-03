# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.coaffiliation`` vs numpy / cupy.

The coaffiliation matrix ``K = C Cᵀ`` (diagonal zeroed) -- the symmetric outer
product of a (soft) community-assignment ``C`` (n x k): ``K_ij`` is the inner
product of nodes i, j's assignment vectors (1 iff they share a community for a
hard one-hot ``C``, the soft overlap otherwise). A nitrix-specific community
primitive with no canonical external library, so the references are the
textbook numpy outer-product floor (+ fp64 oracle) and a CuPy GPU ref. Pure
matmul/broadcast, so GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import coaffiliation

from ._base import BuiltPoint, Case, to_cupy
from ._graph import assignment_input, cupy_coaffiliation, np_coaffiliation

_K = 16  # community count (assignment width)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    C = assignment_input(n, _K, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(C))
    ref = np_coaffiliation(C.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(C)
        return (C,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda c: coaffiliation(c)),
        'numpy.coaffiliation': ('numpy', np_coaffiliation),  # CPU floor
        'cupy.coaffiliation': ('cupy', cupy_coaffiliation()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (nodes): cost ~ n²·k (the dense Gram); k fixed at _K.
_SIZES = [128, 256, 512]

CASE = Case(
    name='coaffiliation',
    op_qualname='nitrix.graph.coaffiliation',
    output_independent=True,  # each K_ij is an independent inner product
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
