# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.pairedcov`` vs numpy / cupy.

The **cross**-covariance between two variable blocks,
``Xc @ Yc^T / (obs - 1)``,
shaped ``(c, d)`` -- a pure BLAS matmul (centring + one gemm), the GPU-friendly
regime (no solver).  System-under-test is nitrix on jax; the CPU floor is the
natural numpy construction and the GPU twin is the same on cupy, scored against
an fp64 oracle.  See ``cases/_conditional_paired.py`` for the shared latent-
factor input (gives a non-trivial cross-block) and the exact convention match
(``ddof=1``, ``rowvar=True``).  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import pairedcov

from ._base import BuiltPoint, Case, to_cupy
from ._conditional_paired import (
    cupy_paired_conditional,
    paired_conditional,
    paired_input,
)

_KIND = 'pairedcov'


def _build(param: Dict[str, Any]) -> BuiltPoint:
    c, d, obs = param['c'], param['d'], param['obs']
    X, Y = paired_input(c, d, obs, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    jy = jax.block_until_ready(jnp.asarray(Y))

    ref = paired_conditional(X.astype(np.float64), Y.astype(np.float64),
                             _KIND, np)  # (c, d) fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, Y)
        return (X, Y) if framework == 'numpy' else (jx, jy)

    baselines = {
        'nitrix-jax': ('jax', lambda x, y: pairedcov(x, y)),
        'numpy.pairedcov': ('numpy',
                            lambda x, y: paired_conditional(x, y, _KIND, np)),
        'cupy.pairedcov': ('cupy', cupy_paired_conditional(_KIND)),  # GPU twin
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (variables_X, variables_Y, observations): c = d (a square cross-block) varied
# to brain-parcel scale; obs > c keeps the per-block variances well-sampled.
_SHAPES = [(128, 128, 1024), (256, 256, 2048), (512, 512, 4096)]
_LARGE = [(1024, 1024, 4096), (2048, 2048, 8192)]  # brain-parcel scale

CASE = Case(
    name='pairedcov',
    op_qualname='nitrix.stats.pairedcov',
    output_independent=False,  # entry (i, j) couples blocks i and j over obs
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'd': d, 'obs': o, 'seed': 0}
                  for (c, d, o) in _SHAPES],
    representative={'c': 256, 'd': 256, 'obs': 2048, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'd': d, 'obs': o, 'seed': 0} for (c, d, o) in _LARGE),
    complexity=(
        'Xc @ Yc^T / (obs - 1): O(c * d * obs) -- one BLAS-class matmul (plus '
        'O((c + d) * obs) centring), the GPU-friendly regime (no solver, '
        'contrast the precision / pca_fit families). HBM ~ (c + d) * obs (the '
        'inputs) + c * d (the cross-block). The size tier varies c = d to '
        'parcel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
