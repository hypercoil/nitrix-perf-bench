# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.pairedcorr`` vs numpy / cupy.

The **cross**-correlation between two variable blocks: ``pairedcov`` divided by
the geometric mean of the two blocks' per-variable variances, shaped ``(c, d)``
-- still solver-free, the GPU-friendly regime.  System-under-test is nitrix on
jax; the CPU floor is the natural numpy construction, the GPU twin the same on
cupy, scored against an fp64 oracle.  See ``cases/_conditional_paired.py`` for
the shared input and the exact normalisation match (``+eps`` outside the
sqrt-of-product, ``ddof=1``).

NOTE (a measured-efficiency angle): nitrix's ``pairedcorr`` forms the **full**
``cov(X)`` (c, c) and ``cov(Y)`` (d, d) just to take their diagonals (the
per-variable variances), so it does ~3x the matmul of the minimal path (one
cross-cov + two variance reductions).  The numpy / cupy floor here computes the
variances directly, so on CPU the floor can lead -- the redundant full-cov is a
candidate nitrix micro-optimisation.  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import pairedcorr

from ._base import BuiltPoint, Case, to_cupy
from ._conditional_paired import (
    cupy_paired_conditional,
    paired_conditional,
    paired_input,
)

_KIND = 'pairedcorr'


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
        'nitrix-jax': ('jax', lambda x, y: pairedcorr(x, y)),
        'numpy.pairedcorr': ('numpy',
                             lambda x, y: paired_conditional(x, y, _KIND, np)),
        'cupy.pairedcorr': ('cupy', cupy_paired_conditional(_KIND)),  # GPU
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (variables_X, variables_Y, observations): c = d varied to brain-parcel scale;
# obs > c keeps the per-block variances (the normalisation) well-sampled.
_SHAPES = [(128, 128, 1024), (256, 256, 2048), (512, 512, 4096)]
_LARGE = [(1024, 1024, 4096), (2048, 2048, 8192)]  # brain-parcel scale

CASE = Case(
    name='pairedcorr',
    op_qualname='nitrix.stats.pairedcorr',
    output_independent=False,  # cross-block + a per-block variance normaliser
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'd': d, 'obs': o, 'seed': 0}
                  for (c, d, o) in _SHAPES],
    representative={'c': 256, 'd': 256, 'obs': 2048, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'd': d, 'obs': o, 'seed': 0} for (c, d, o) in _LARGE),
    complexity=(
        'cross-cov O(c * d * obs) then a geometric-mean normalisation. nitrix '
        'forms the full cov(X) O(c^2 * obs) and cov(Y) O(d^2 * obs) to take '
        'their diagonals -- ~3x the minimal matmul (same complexity class); '
        'the floor computes the variances directly. Solver-free (GPU). '
        'HBM ~ (c + d) * obs. The size tier varies c = d to parcel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
