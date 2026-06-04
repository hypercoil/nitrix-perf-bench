# -*- coding: utf-8 -*-
"""Tier-2 numerics: ``nitrix.numerics.zscore_normalize`` vs scipy / cupy.

Z-score normalisation ``(x - mean)/(std + eps)`` over the trailing axis
(population std, ddof=0) -- nitrix (jax) vs **``scipy.stats.zscore``** (the
canonical domain reference; ddof=0 default, matches to ~1e-12 in fp64 -- the
eps) + a CuPy GPU ref, scored against an fp64 oracle. Memory-bound
reduction, GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.numerics import zscore_normalize

from ._base import BuiltPoint, Case, to_cupy
from ._normalize import cupy_zscore, normalize_input, np_zscore, scipy_zscore


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    X = normalize_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_zscore(X.astype('float64'))  # fp64 oracle (nitrix's eps)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: zscore_normalize(x)),
        'scipy.stats.zscore': ('scipy', scipy_zscore()),  # CPU floor
        'cupy.zscore_normalize': ('cupy', cupy_zscore()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (rows = cols): reduce over the trailing axis; cost ~ n².
_SIZES = [512, 2048, 4096]

CASE = Case(
    name='zscore_normalize',
    op_qualname='nitrix.numerics.zscore_normalize',
    output_independent=False,  # each row shares its mean / std reduction
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
