# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.conditionalcorr`` vs numpy / cupy.

Correlation of ``X`` **after residualising out** the ``Y`` subspace: the
normalised ``conditionalcov`` (each entry divided by the geometric mean of the
residual variances), shaped ``(c, c)`` -- the confound-adjusted connectome.
System-under-test is nitrix on jax; the CPU floor is the natural numpy
construction, the GPU twin the same on cupy, scored against an fp64 oracle.
See ``cases/_conditional_paired.py`` for the shared input and the exact
convention
match (the ``_corrnorm`` normalisation, ``ddof=1``, no-intercept residualise).

GPU note: as ``conditionalcov`` -- the only solver is the tiny ``(d, d)``
confound-Gram factorisation; the op is matmul-dominated and GPU-robust.  Ratio
vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import conditionalcorr

from ._base import BuiltPoint, Case, to_cupy
from ._conditional_paired import (
    cupy_paired_conditional,
    paired_conditional,
    paired_input,
)

_KIND = 'conditionalcorr'


def _build(param: Dict[str, Any]) -> BuiltPoint:
    c, d, obs = param['c'], param['d'], param['obs']
    X, Y = paired_input(c, d, obs, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    jy = jax.block_until_ready(jnp.asarray(Y))

    ref = paired_conditional(X.astype(np.float64), Y.astype(np.float64),
                             _KIND, np)  # (c, c) fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, Y)
        return (X, Y) if framework == 'numpy' else (jx, jy)

    baselines = {
        'nitrix-jax': ('jax', lambda x, y: conditionalcorr(x, y)),
        'numpy.conditionalcorr': (
            'numpy', lambda x, y: paired_conditional(x, y, _KIND, np)),
        'cupy.conditionalcorr': ('cupy', cupy_paired_conditional(_KIND)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (signal_variables, confounds, observations): c = the signal block (parcels)
# varied to brain scale; d = a handful of nuisance regressors; obs > c.
_SHAPES = [(128, 16, 1024), (256, 16, 2048), (512, 16, 4096)]
_LARGE = [(1024, 16, 4096), (2048, 32, 8192)]  # brain-parcel scale

CASE = Case(
    name='conditionalcorr',
    op_qualname='nitrix.stats.conditionalcorr',
    output_independent=False,  # residualisation + cov + normalise couple all
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'd': d, 'obs': o, 'seed': 0}
                  for (c, d, o) in _SHAPES],
    representative={'c': 256, 'd': 16, 'obs': 2048, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'd': d, 'obs': o, 'seed': 0} for (c, d, o) in _LARGE),
    complexity=(
        'residualise (OLS: a (d, d) Gram + Cholesky O(d^3) + projection '
        'O(c * obs * d)) then cov O(c^2 * obs) then a geometric-mean '
        'normalisation. The cov dominates at parcel c, so matmul-bound and '
        'GPU-robust -- the (d, d) solver is tiny (contrast pca_fit). HBM ~ '
        'c * obs (input) + c^2 (output). The size tier varies c to parcel '
        'scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
