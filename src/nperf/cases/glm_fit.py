# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.glm_fit`` vs nilearn / statsmodels / numpy.

The mass-univariate OLS GLM: fit ``Y[V, N]`` (V voxels x N observations) on a
shared design ``X[N, p]`` -> per-voxel betas ``coef[V, p]``.  THE neuroimaging
fit; the perf story is nitrix vmapping all V voxels behind one fit vs the
community tools.  Baselines: **nilearn** ``run_glm`` (vectorised -- the neuro
mass-univariate standard), **statsmodels** per-voxel ``OLS`` loop (the slow
reference -- declared slow, worker-timeout-capped at large V), and an exact
**numpy** OLS fp64 oracle (lstsq betas; nitrix matches ~7e-7).  Keyed ``V`` =
voxels (the scale axis), ``N`` = observations, ``p`` = regressors.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import glm_fit

from ._base import BuiltPoint, Case, SlowBaseline
from ._glm import glm_data, nilearn_glm_beta, np_glm_beta, statsmodels_glm_beta


def _build(param: Dict[str, Any]) -> BuiltPoint:
    V, N, p = int(param['V']), int(param['N']), int(param['p'])
    Y, X = glm_data(V, N, p, param.get('seed', 0))
    jy = jax.block_until_ready(jnp.asarray(Y))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_glm_beta()(Y, X)  # numpy OLS betas (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jy, jx) if framework == 'jax' else (Y, X)

    baselines = {
        'nitrix-jax': ('jax', lambda Y, X: glm_fit(Y, X).coef),
        'nilearn.run_glm': ('numpy', nilearn_glm_beta()),       # vectorised
        'statsmodels.OLS': ('numpy', statsmodels_glm_beta()),  # loop
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (voxels V, observations N, regressors p): the mass-univariate scale is V.
_SHAPES = [(1000, 120, 8), (10000, 120, 8), (50000, 120, 8)]
_LARGE = [(100000, 120, 8), (200000, 200, 12)]

CASE = Case(
    name='glm_fit',
    op_qualname='nitrix.stats.glm_fit',
    output_independent=True,   # each voxel's OLS fit is independent
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'N': n, 'p': p, 'seed': 0}
                  for (v, n, p) in _SHAPES],
    representative={'V': 10000, 'N': 120, 'p': 8, 'seed': 0},
    large_param_points=tuple(
        {'V': v, 'N': n, 'p': p, 'seed': 0} for (v, n, p) in _LARGE),
    slow_baselines=(
        SlowBaseline('statsmodels.OLS',
                     reason='statsmodels fits each voxel in a python loop '
                            '(O(V) separate OLS) -> minutes at large V; '
                            'declared slow (worker-timeout-capped). nilearn + '
                            'nitrix are vectorised.'),),
    complexity=(
        'one shared (X^T X)^-1 X^T applied to all V voxels: O(V * N * p) '
        '(BLAS-class), nitrix vmaps it behind one compile. nilearn vectorises '
        'too; statsmodels loops V separate fits (the gap grows with V). HBM ~ '
        'V*N (the data) + V*p (the betas). The size tier grows V to '
        'whole-brain voxel counts.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
