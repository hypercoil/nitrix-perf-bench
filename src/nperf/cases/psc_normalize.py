# -*- coding: utf-8 -*-
"""Tier-2 numerics: ``nitrix.numerics.psc_normalize`` vs numpy / cupy.

Percent signal change ``100·(x - mean)/(mean + eps)`` over the trailing axis --
the fMRI BOLD convention -- nitrix (jax) vs the numpy reimplementation (CPU
floor + fp64 oracle) + a CuPy GPU ref. A domain convention (no single library
function), so the numpy reimplementation carrying nitrix's eps is the oracle.
Memory-bound reduction, GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.numerics import psc_normalize

from ._base import BuiltPoint, Case, to_cupy
from ._normalize import cupy_psc, normalize_input, np_psc


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    X = normalize_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_psc(X.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: psc_normalize(x)),
        'numpy.psc': ('numpy', np_psc),  # CPU floor
        'cupy.psc_normalize': ('cupy', cupy_psc()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (rows = cols): reduce over the trailing axis; cost ~ n².
_SIZES = [512, 2048, 4096]

CASE = Case(
    name='psc_normalize',
    op_qualname='nitrix.numerics.psc_normalize',
    output_independent=False,  # each row shares its mean reduction
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
