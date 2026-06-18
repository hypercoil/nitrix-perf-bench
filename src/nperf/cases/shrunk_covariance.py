# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.shrunk_covariance`` vs scikit-learn.

The ``method``-dispatched analytic-shrinkage covariance (the cov only, no
intensity returned).  Benchmarked at the default ``method='ledoit_wolf'``, so
**scikit-learn's ``ledoit_wolf`` cov is the fp64 oracle + community baseline**
(nitrix matches ~4e-7).  Distinct public op from ``ledoit_wolf`` (the dispatch
entry point); same ``X[n, p]`` -> ``cov[p, p]`` cost.  No on-device twin -> GPU
headline nitrix-jax vs the sklearn CPU bar.  Keyed ``c`` = features, ``n_obs``
= samples.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import shrunk_covariance

from ._base import BuiltPoint, Case
from ._shrinkage import shrinkage_data, sk_ledoit_wolf


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p, n = int(param['c']), int(param['n_obs'])
    X = shrinkage_data(n, p, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = sk_ledoit_wolf(fp64=True)(X)  # method='ledoit_wolf' -> LW cov

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: shrunk_covariance(x)),
        'sklearn.ledoit_wolf': ('numpy', sk_ledoit_wolf()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(200, 100), (500, 200), (1000, 400)]
_LARGE = [(2000, 500), (4000, 300)]

CASE = Case(
    name='shrunk_covariance',
    op_qualname='nitrix.stats.shrunk_covariance',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _SHAPES],
    representative={'c': 500, 'n_obs': 200, 'seed': 0},
    large_param_points=tuple(
        {'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _LARGE),
    complexity=(
        'the analytic-shrinkage cov at the default method (Ledoit-Wolf): '
        'centred Gram (O(p^2 n)) + the closed-form intensity; BLAS-class, HBM '
        '~ p^2. The dispatcher entry point (cov only). sklearn LW is the CPU '
        'bar.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
