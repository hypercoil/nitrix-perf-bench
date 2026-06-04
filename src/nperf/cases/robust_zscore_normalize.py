# -*- coding: utf-8 -*-
"""Tier-2 numerics: ``nitrix.numerics.robust_zscore_normalize``.

vs numpy / cupy. Robust z-score ``(x - median)/(1.4826·MAD + eps)`` over the
trailing axis (median + median-absolute-deviation; outlier-resistant) -- nitrix
(jax) vs the numpy reimplementation (CPU floor + fp64 oracle) + a CuPy GPU ref.
nitrix uses the truncated literal ``1.4826``;
``scipy.stats.median_abs_deviation(scale='normal')`` uses the full
``1/Phi^-1(0.75) = 1.48260222`` (a ~1.5e-6 relative difference), so the oracle
here matches nitrix's exact constant rather than scipy's. Median-based
reduction, GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.numerics import robust_zscore_normalize

from ._base import BuiltPoint, Case, to_cupy
from ._normalize import cupy_robust, normalize_input, np_robust


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    X = normalize_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_robust(X.astype('float64'))  # fp64 oracle (nitrix's 1.4826)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: robust_zscore_normalize(x)),
        'numpy.robust_zscore': ('numpy', np_robust),  # CPU floor
        'cupy.robust_zscore_normalize': ('cupy', cupy_robust()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (rows = cols): per-row median + MAD over the trailing axis; cost ~ n² log n.
_SIZES = [512, 2048, 4096]

CASE = Case(
    name='robust_zscore_normalize',
    op_qualname='nitrix.numerics.robust_zscore_normalize',
    output_independent=False,  # each row shares its median / MAD reduction
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
