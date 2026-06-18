# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.oas`` vs scikit-learn.

Oracle-Approximating-Shrinkage covariance (Chen et al. 2010) -- the sibling of
Ledoit-Wolf with a different closed-form intensity (better under Gaussianity).
``X[n, p]`` -> ``(cov[p, p], shrinkage)``; the baseline reads ``[0]``.
**scikit-learn's ``oas`` is the reference implementation** = community baseline
+ fp64 oracle (nitrix matches the cov ~5e-7, intensity bit-for-bit).  No
on-device twin -> GPU headline nitrix-jax vs the sklearn CPU bar.  Keyed
``c`` = features (the ``c^2`` axis), ``n_obs`` = samples.  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import oas

from ._base import BuiltPoint, Case
from ._shrinkage import shrinkage_data, sk_oas


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p, n = int(param['c']), int(param['n_obs'])
    X = shrinkage_data(n, p, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = sk_oas(fp64=True)(X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: oas(x)[0]),
        'sklearn.oas': ('numpy', sk_oas()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(200, 100), (500, 200), (1000, 400)]
_LARGE = [(2000, 500), (4000, 300)]

CASE = Case(
    name='oas',
    op_qualname='nitrix.stats.oas',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _SHAPES],
    representative={'c': 500, 'n_obs': 200, 'seed': 0},
    large_param_points=tuple(
        {'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _LARGE),
    complexity=(
        'as ledoit_wolf -- centred Gram (O(p^2 n)) + the closed-form OAS '
        'intensity (p^2 reductions); BLAS-class, HBM ~ p^2. The size tier '
        'grows p into the p>n regime; sklearn (reference impl) is the CPU '
        'bar.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
