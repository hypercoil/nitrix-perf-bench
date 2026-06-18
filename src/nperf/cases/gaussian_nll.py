# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.gaussian_nll`` vs numpy / scipy / cupy.

Negative log-likelihood of ``x`` under ``N(mean, exp(log_var))`` -- the VAE /
probabilistic-decoder reconstruction loss.  Elementwise
``0.5(log 2pi + log_var + (x-mean)^2 / exp(log_var))`` then a mean reduction.
Three references: an exact-formula **numpy** fp64 oracle, **scipy.stats.norm**
(the same NLL via ``-mean(norm.logpdf)`` -- a community cross-check), and a
**cupy** reimplementation as the apples-to-apples GPU bar (this is the
elementwise + reduce regime where cupy is the right on-device comparison).
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import gaussian_nll

from ._base import BuiltPoint, Case, to_cupy
from ._gaussian import (
    cupy_gaussian_nll,
    gaussian_inputs,
    np_gaussian_nll,
    scipy_gaussian_nll,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    x, mean, log_var = gaussian_inputs(shape, param.get('seed', 0))
    jx = tuple(jax.block_until_ready(jnp.asarray(a))
               for a in (x, mean, log_var))
    ref = np_gaussian_nll(fp64=True)(x, mean, log_var)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(x, mean, log_var)
        if framework == 'jax':
            return jx
        return (x, mean, log_var)

    baselines = {
        'nitrix-jax': ('jax', lambda x, m, lv: gaussian_nll(x, m, lv)),
        'numpy.gaussian_nll': ('numpy', np_gaussian_nll()),  # exact formula
        'scipy.norm_nll': ('numpy', scipy_gaussian_nll()),   # community
        'cupy.gaussian_nll': ('cupy', cupy_gaussian_nll()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (batch, latent dim): the VAE regime -- elementwise + a global mean reduce.
_SHAPES = [[4096, 64], [16384, 128], [65536, 256]]
_LARGE = [[262144, 256], [262144, 512]]

CASE = Case(
    name='gaussian_nll',
    op_qualname='nitrix.stats.gaussian_nll',
    output_independent=False,  # the mean reduction couples all elements
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [16384, 128], 'seed': 0},
    large_param_points=tuple({'shape': s, 'seed': 0} for s in _LARGE),
    complexity=(
        'elementwise (exp + a squared residual) + a global mean reduce: '
        'O(N) memory-bound, the fused-XLA-vs-cupy-multi-kernel regime. HBM ~ '
        '3N (x, mean, log_var). The size tier grows the batch.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
