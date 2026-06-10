# -*- coding: utf-8 -*-
"""Morphology pooling: ``nitrix.morphology.max_pool_with_indices_nd``.

N-D max pool returning ``(values, indices)`` over a ``(B, C, *spatial)`` field
(non-overlapping 2³ blocks, 3-D).  Benched as the **with-indices** op: the
argmax is real work -- on the L4 ~2.6× a max-only pool (0.85 vs 0.32 ms at
128³) -- so the nitrix baseline keeps the indices **live** with ``1e-30*idx``.
(Returning just ``values`` lets XLA DCE the argmax, and ``0*idx`` is folded
away, both defeating the measurement; the references compute the argmax
eagerly, so all three do the same work -- a fair comparison.)

Warranted: the values are the exact windowed max -- a numpy windowed-max fp64
oracle + CPU floor (verified exact), cupy the GPU bar.  The argmax index
convention (global flat spatial index) + the round-trip are pinned via
``max_unpool_nd`` in the tests.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import max_pool_with_indices_nd

from ._base import BuiltPoint, Case, to_cupy
from ._pooling import cupy_pool, np_pool, pool_input


def _nx_pool(x: Any) -> Any:
    # keep the argmax live (XLA DCEs an unused [1]; 0*idx is folded away).
    m, i = max_pool_with_indices_nd(x, pool_size=2, spatial_rank=3)
    return m + i.astype(m.dtype) * 1e-30


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    x = pool_input(d, param.get('seed', 0))
    xj = jax.block_until_ready(jnp.asarray(x))
    ref = np_pool(x.astype('float64'))  # fp64 oracle (windowed max)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(x)
        return (x,) if framework == 'numpy' else (xj,)

    baselines = {
        'nitrix-jax': ('jax', _nx_pool),
        'numpy.max_pool': ('numpy', np_pool),
        'cupy.max_pool': ('cupy', cupy_pool()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='max_pool_with_indices_nd',
    op_qualname='nitrix.morphology.max_pool_with_indices_nd',
    output_independent=True,  # each pooled value = max of its own window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the input voxel count N (B*C*d^3): one windowed max + a '
        'windowed argmax per non-overlapping 2^3 block. Embarrassingly '
        'parallel, memory-bandwidth-bound, GPU-pure; the argmax ~doubles the '
        'cost over a max-only pool (~2.6x on the L4). The tier varies N.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
