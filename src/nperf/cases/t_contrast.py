# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.t_contrast`` vs nilearn / statsmodels / numpy.

The mass-univariate t-test: the realistic inference pipeline -- fit the GLM at
every voxel then read a t-contrast (here the last regressor) -> ``t[V]``.  The
baseline times the FULL fit+contrast (what a user actually runs), so it is the
end-to-end mass-univariate t-map workload.  Baselines: **nilearn** (``run_glm``
+ ``compute_contrast`` -- vectorised), **statsmodels** (per-voxel ``OLS`` +
``t_test`` loop -- slow), and an exact **numpy** OLS t-statistic fp64 oracle
(nitrix matches ~1e-5).  Keyed ``V`` = voxels (scale axis).  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import glm_fit, t_contrast

from ._base import BuiltPoint, Case, SlowBaseline
from ._glm import glm_data, nilearn_contrast, np_t_stat, statsmodels_contrast


def _build(param: Dict[str, Any]) -> BuiltPoint:
    V, N, p = int(param['V']), int(param['N']), int(param['p'])
    Y, X = glm_data(V, N, p, param.get('seed', 0))
    c = np.eye(p, dtype=np.float32)[-1]          # the last regressor's effect
    jy = jax.block_until_ready(jnp.asarray(Y))
    jx = jax.block_until_ready(jnp.asarray(X))
    jc = jnp.asarray(c)
    ref = np_t_stat(c)(Y, X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jy, jx) if framework == 'jax' else (Y, X)

    baselines = {
        # full pipeline: fit then read the t-contrast (index 2 = the t-stat).
        'nitrix-jax': ('jax',
                       lambda Y, X: t_contrast(glm_fit(Y, X), jc)[2]),
        'nilearn.compute_contrast': ('numpy', nilearn_contrast(c, 't')),
        'statsmodels.t_test': ('numpy', statsmodels_contrast(c, 't')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(1000, 120, 8), (10000, 120, 8), (50000, 120, 8)]
_LARGE = [(100000, 120, 8), (200000, 200, 12)]

CASE = Case(
    name='t_contrast',
    op_qualname='nitrix.stats.t_contrast',
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'N': n, 'p': p, 'seed': 0}
                  for (v, n, p) in _SHAPES],
    representative={'V': 10000, 'N': 120, 'p': 8, 'seed': 0},
    large_param_points=tuple(
        {'V': v, 'N': n, 'p': p, 'seed': 0} for (v, n, p) in _LARGE),
    slow_baselines=(
        SlowBaseline('statsmodels.t_test',
                     reason='statsmodels loops V voxels (OLS+t_test) -> '
                            'minutes at large V; declared slow.'),),
    complexity=(
        'the end-to-end mass-univariate t-map: a shared OLS fit + a cheap '
        'per-voxel t = c.beta / se. nitrix vmaps the whole pipeline; '
        'nilearn vectorises; statsmodels loops. HBM ~ V*N. Size tier grows '
        'V; the fit dominates the contrast.'),
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
