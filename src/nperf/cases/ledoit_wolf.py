# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.ledoit_wolf`` vs scikit-learn.

Ledoit-Wolf analytic-shrinkage covariance -- the canonical high-dimensional
estimator: ``X[n, p]`` -> a regularised ``cov[p, p]`` shrunk toward a scaled
identity by a closed-form optimal intensity.  **scikit-learn's ``ledoit_wolf``
is the reference implementation**, so it is BOTH the community baseline and the
fp64 oracle (nitrix matches its cov to ~4e-7, shrinkage intensity bit-for-bit).
nitrix returns ``(cov, shrinkage)``; the baseline reads ``[0]`` (the cov --
forces the full estimate).  No on-device twin (sklearn is CPU-only), so the GPU
headline is nitrix-jax (jax-cuda12) vs the sklearn CPU bar.  Keyed ``c`` =
features (= cov dim, the ``c^2`` scale/HBM axis), ``n_obs`` = samples; the size
tier pushes into the ``p > n`` regime where shrinkage earns its keep.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import ledoit_wolf

from ._base import BuiltPoint, Case
from ._shrinkage import shrinkage_data, sk_ledoit_wolf


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p, n = int(param['c']), int(param['n_obs'])
    X = shrinkage_data(n, p, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = sk_ledoit_wolf(fp64=True)(X)  # the canonical impl, in fp64 = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: ledoit_wolf(x)[0]),
        'sklearn.ledoit_wolf': ('numpy', sk_ledoit_wolf()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (features p, samples n): the p >~ n high-dim regime (connectome parcels x
# subjects) where analytic shrinkage matters.
_SHAPES = [(200, 100), (500, 200), (1000, 400)]
_LARGE = [(2000, 500), (4000, 300)]

CASE = Case(
    name='ledoit_wolf',
    op_qualname='nitrix.stats.ledoit_wolf',
    output_independent=False,  # cov[i,j] couples features i,j over all samples
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _SHAPES],
    representative={'c': 500, 'n_obs': 200, 'seed': 0},
    large_param_points=tuple(
        {'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _LARGE),
    complexity=(
        'centred Gram X@X.T (O(p^2 n)) + the closed-form LW intensity (a few '
        'p^2 reductions over the sample cross-products) -- BLAS-class, the '
        'GPU-friendly regime. HBM ~ p^2 (the cov). The size tier grows p into '
        'the p>n regime; sklearn (the reference impl) is the CPU bar.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
