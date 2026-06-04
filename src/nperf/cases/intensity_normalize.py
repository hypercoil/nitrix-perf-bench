# -*- coding: utf-8 -*-
"""Tier-2 numerics: ``nitrix.numerics.intensity_normalize`` vs numpy / cupy.

Percentile-clip to ``[p1, p99]`` then rescale to ``[0, 1]`` over the whole
tensor (``axis=None``) -- the synthstrip / SynthSeg pre-training convention --
nitrix (jax) vs the numpy reimplementation (percentile clip + rescale; CPU
floor + fp64 oracle) + a CuPy GPU ref. The percentile method matches (both
linear), verified equal to 0.0 in fp64. Memory-bound, GPU-pure. Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.numerics import intensity_normalize

from ._base import BuiltPoint, Case, to_cupy
from ._normalize import cupy_intensity, normalize_input, np_intensity


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    X = normalize_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_intensity(X.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: intensity_normalize(x)),
        'numpy.intensity': ('numpy', np_intensity),  # CPU floor
        'cupy.intensity_normalize': ('cupy', cupy_intensity()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (rows = cols): percentiles over the whole tensor; cost ~ n² (+ a sort).
_SIZES = [512, 2048, 4096]

CASE = Case(
    name='intensity_normalize',
    op_qualname='nitrix.numerics.intensity_normalize',
    output_independent=False,  # global percentiles couple every element
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
