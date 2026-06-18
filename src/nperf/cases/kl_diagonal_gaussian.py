# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.kl_diagonal_gaussian`` vs numpy / cupy.

KL divergence of ``N(mean, diag exp(log_var))`` from the standard normal
``N(0, I)`` -- the VAE latent regulariser.  Closed form
``0.5 sum(exp(log_var) + mean^2 - 1 - log_var)`` (a sum reduction); elementwise
+ a global reduce.  References: an exact-formula **numpy** fp64 oracle and a
**cupy** reimplementation as the apples-to-apples GPU bar (no single community
function for this analytic KL -- it is a numpy/cupy formula).  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import kl_diagonal_gaussian

from ._base import BuiltPoint, Case, to_cupy
from ._gaussian import cupy_kl_diag, gaussian_inputs, np_kl_diag


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    _, mean, log_var = gaussian_inputs(shape, param.get('seed', 0))
    jm = jax.block_until_ready(jnp.asarray(mean))
    jlv = jax.block_until_ready(jnp.asarray(log_var))
    ref = np_kl_diag(fp64=True)(mean, log_var)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mean, log_var)
        if framework == 'jax':
            return (jm, jlv)
        return (mean, log_var)

    baselines = {
        'nitrix-jax': ('jax', lambda m, lv: kl_diagonal_gaussian(m, lv)),
        'numpy.kl_diagonal_gaussian': ('numpy', np_kl_diag()),  # exact formula
        'cupy.kl_diagonal_gaussian': ('cupy', cupy_kl_diag()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[4096, 64], [16384, 128], [65536, 256]]
_LARGE = [[262144, 256], [262144, 512]]

CASE = Case(
    name='kl_diagonal_gaussian',
    op_qualname='nitrix.stats.kl_diagonal_gaussian',
    output_independent=False,  # the sum reduction couples all elements
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [16384, 128], 'seed': 0},
    large_param_points=tuple({'shape': s, 'seed': 0} for s in _LARGE),
    complexity=(
        'elementwise (exp + square) + a global sum reduce: O(N) memory-bound. '
        'HBM ~ 2N (mean, log_var). The size tier grows the batch.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
