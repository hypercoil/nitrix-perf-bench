# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.ebic_score`` vs a numpy oracle.

Extended BIC of a precision estimate (Foygel & Drton 2010) -- the glasso-path
model-selection criterion: ``n (tr(S Theta) - logdet Theta) + E log n + 4 gamma
E log p`` (``E`` = off-diagonal edge count).  A cheap scalar reduction (one
logdet + a couple of ``p^2`` sums); there is no single community function for
it, so the reference is an **exact numpy reimplementation of the same formula**
(fp64 oracle; ``numpy.*`` -> not in the CPU-community lens).  The inputs are a
real sparse precision (a sklearn graphical-LASSO solution) + its empirical
covariance.  No on-device twin -> GPU headline nitrix-jax vs the numpy CPU bar.
Keyed ``c`` = ``p``, ``n_obs`` = the EBIC sample count ``n``.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import ebic_score

from ._base import BuiltPoint, Case
from ._shrinkage import np_ebic, sk_glasso, sparse_precision_cov

_LAM = 0.1
_GAMMA = 0.5


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p, n = int(param['c']), int(param['n_obs'])
    S = sparse_precision_cov(p, param.get('seed', 0))  # well-conditioned
    theta = sk_glasso(_LAM, fp64=True)(S)  # a real sparse precision
    sj = jax.block_until_ready(jnp.asarray(S))
    tj = jax.block_until_ready(jnp.asarray(theta))
    ref = np_ebic(n, _GAMMA)(theta, S)  # exact-formula numpy oracle (fp64)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (tj, sj) if framework == 'jax' else (theta, S)

    baselines = {
        'nitrix-jax': ('jax',
                       lambda th, s: ebic_score(th, s, n, gamma=_GAMMA)),
        'numpy.ebic_score': ('numpy', np_ebic(n, _GAMMA)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [(80, 160), (160, 320), (320, 640)]   # (p, n)
_LARGE = [(640, 1280), (1000, 2000)]

CASE = Case(
    name='ebic_score',
    op_qualname='nitrix.stats.ebic_score',
    output_independent=False,  # logdet + global reductions over the p x p
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _SIZES],
    representative={'c': 160, 'n_obs': 320, 'seed': 0},
    large_param_points=tuple(
        {'c': p, 'n_obs': n, 'seed': 0} for (p, n) in _LARGE),
    complexity=(
        'a scalar: one rolled-Cholesky logdet (O(p^3), cuSOLVER-free) + two '
        'p^2 reductions (tr(S Theta), the edge count). Cheap; the numpy '
        'same-formula oracle is the CPU bar (no single community fn). HBM ~ '
        'p^2. The size tier grows p.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
