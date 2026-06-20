# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.signal.product_filter`` vs numpy / cupy.

Frequency-domain convolution: ``irfft(weight * rfft(X))`` -- a circular
convolution along the time axis with a freq-domain ``weight`` (rfft length
``t//2+1``).  The natural impl is rfft multiplication, so the reference is an
exact **numpy** ``np.fft`` reimplementation (fp64 oracle) + a **cupy**
``cp.fft`` reimpl as the apples-to-apples GPU bar (FFT-based, GPU-pure).  Both
get the SAME signal + weight.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.signal import product_filter

from ._base import BuiltPoint, Case, to_cupy
from ._signal import (
    cupy_product_filter,
    freq_weight,
    np_product_filter,
    signal_input,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = int(param['n_sig']), int(param['t'])
    X = signal_input(n_sig, t, param.get('seed', 0))
    W = freq_weight(t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    jw = jax.block_until_ready(jnp.asarray(W))
    ref = np_product_filter(fp64=True)(X, W)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, W)
        return (jx, jw) if framework == 'jax' else (X, W)

    baselines = {
        'nitrix-jax': ('jax', lambda x, w: product_filter(x, w)),
        'numpy.product_filter': ('numpy', np_product_filter()),
        'cupy.product_filter': ('cupy', cupy_product_filter()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(64, 4096), (256, 8192), (1024, 8192)]
_LARGE = [(2048, 16384), (4096, 16384)]

CASE = Case(
    name='product_filter',
    op_qualname='nitrix.signal.product_filter',
    output_independent=False,  # the FFT couples all timepoints
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 256, 't': 8192, 'seed': 0},
    large_param_points=tuple(
        {'n_sig': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'rfft + a complex elementwise multiply + irfft: O(N log t), GPU-pure '
        '(a clean cupy bar). HBM ~ batch*t. The size tier grows batch x '
        'length.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
