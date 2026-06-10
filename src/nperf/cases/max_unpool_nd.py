# -*- coding: utf-8 -*-
"""Morphology pooling: ``nitrix.morphology.max_unpool_nd``.

Scatter pooled values back into a higher-resolution zeroed grid at their argmax
indices -- the inverse of ``max_pool_with_indices_nd`` (the unpool half of a
max-pool / unpool U-Net pair).  The ``(values, indices)`` input is the real
output of nitrix ``max_pool_with_indices_nd`` on a random field, so the indices
are valid in nitrix's flat convention.

Warranted: the numpy reimplementation scatters the same values to the same flat
indices (verified exact vs nitrix) -- the fp64 oracle + CPU floor; cupy is the
GPU bar.  A pure scatter (gather/scatter-bound), GPU-pure.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import max_unpool_nd

from ._base import BuiltPoint, Case, to_cupy
from ._pooling import cupy_unpool, np_unpool, pool_for_unpool


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    out_shape = [d, d, d]
    values, indices = pool_for_unpool(d, param.get('seed', 0))
    vj = jax.block_until_ready(jnp.asarray(values))
    ij = jax.block_until_ready(jnp.asarray(indices))
    ref = np_unpool(out_shape)(values.astype('float64'), indices)  # oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(values, indices)
        return (values, indices) if framework == 'numpy' else (vj, ij)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda v, i: max_unpool_nd(v, i, output_shape=out_shape,
                                       spatial_rank=3)),
        'numpy.max_unpool': ('numpy', np_unpool(out_shape)),
        'cupy.max_unpool': ('cupy', cupy_unpool(out_shape)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='max_unpool_nd',
    op_qualname='nitrix.morphology.max_unpool_nd',
    output_independent=False,  # a scatter -- a window's target is data-dep
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the output voxel count N (B*C*d^3): allocate a zeroed grid '
        'and scatter one value per pooled position to its flat index. A '
        'scatter (data-dependent writes), memory-bandwidth-bound, GPU-pure; '
        'HBM ~ N. The size tier varies the output volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
