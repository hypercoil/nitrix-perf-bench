# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.f_contrast`` vs nilearn / statsmodels / numpy.

The mass-univariate F-test: fit the GLM at every voxel then read an F-contrast
(here the joint effect of the last two regressors, ``m=2``) -> ``F[V]``.  The
baseline times the FULL fit+contrast (the end-to-end mass-univariate F-map).
Baselines: **nilearn** (``run_glm`` + ``compute_contrast`` F -- vectorised),
**statsmodels** (per-voxel ``OLS`` + ``f_test`` loop -- slow), and an exact
**numpy** OLS F-statistic fp64 oracle (nitrix matches ~1e-6 rel).  Keyed ``V``
= voxels (scale axis).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import f_contrast, glm_fit

from ._base import BuiltPoint, Case, SlowBaseline
from ._glm import glm_data, nilearn_contrast, np_f_stat, statsmodels_contrast


def _build(param: Dict[str, Any]) -> BuiltPoint:
    V, N, p = int(param['V']), int(param['N']), int(param['p'])
    Y, X = glm_data(V, N, p, param.get('seed', 0))
    C = np.eye(p, dtype=np.float32)[-2:]         # joint effect of last 2 regs
    jy = jax.block_until_ready(jnp.asarray(Y))
    jx = jax.block_until_ready(jnp.asarray(X))
    jC = jnp.asarray(C)
    ref = np_f_stat(C)(Y, X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jy, jx) if framework == 'jax' else (Y, X)

    baselines = {
        # full pipeline: fit then read the F-contrast (index 0 = the F-stat).
        'nitrix-jax': ('jax',
                       lambda Y, X: f_contrast(glm_fit(Y, X), jC)[0]),
        'nilearn.compute_contrast': ('numpy', nilearn_contrast(C, 'F')),
        'statsmodels.f_test': ('numpy', statsmodels_contrast(C, 'F')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(1000, 120, 8), (10000, 120, 8), (50000, 120, 8)]
_LARGE = [(100000, 120, 8), (200000, 200, 12)]

CASE = Case(
    name='f_contrast',
    op_qualname='nitrix.stats.f_contrast',
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'N': n, 'p': p, 'seed': 0}
                  for (v, n, p) in _SHAPES],
    representative={'V': 10000, 'N': 120, 'p': 8, 'seed': 0},
    large_param_points=tuple(
        {'V': v, 'N': n, 'p': p, 'seed': 0} for (v, n, p) in _LARGE),
    slow_baselines=(
        SlowBaseline('statsmodels.f_test',
                     reason='statsmodels loops V voxels (OLS+f_test) -> '
                            'minutes at large V; declared slow.'),),
    complexity=(
        'the end-to-end mass-univariate F-map: a shared OLS fit + a per-voxel '
        'F (m x m inverse, m=2). nitrix vmaps the pipeline; nilearn '
        'vectorises; statsmodels loops. HBM ~ V*N. The size tier grows V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
