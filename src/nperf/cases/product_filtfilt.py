# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.product_filtfilt`` vs numpy / cupy.

Zero-phase forward-backward frequency-domain filter: ``product_filter``, flip,
``product_filter`` again, flip back (net zero phase delay; amplitude response
quadratic in ``weight``).  Reference: an exact **numpy** ``np.fft`` reimpl of
the same forward-backward sequence (fp64 oracle) + a **cupy** reimpl as the
GPU bar (FFT-based, GPU-pure).  Both get the SAME signal + weight.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import product_filtfilt

from ._base import BuiltPoint, Case, to_cupy
from ._signal import (
    cupy_product_filter,
    freq_weight,
    np_product_filtfilt,
    signal_input,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = int(param['n_sig']), int(param['t'])
    X = signal_input(n_sig, t, param.get('seed', 0))
    W = freq_weight(t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    jw = jax.block_until_ready(jnp.asarray(W))
    ref = np_product_filtfilt(fp64=True)(X, W)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, W)
        return (jx, jw) if framework == 'jax' else (X, W)

    baselines = {
        'nitrix-jax': ('jax', lambda x, w: product_filtfilt(x, w)),
        'numpy.product_filtfilt': ('numpy', np_product_filtfilt()),
        'cupy.product_filtfilt': ('cupy', cupy_product_filter(filtfilt=True)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(64, 4096), (256, 8192), (1024, 8192)]
_LARGE = [(2048, 16384), (4096, 16384)]

CASE = Case(
    name='product_filtfilt',
    op_qualname='nitrix.stats.product_filtfilt',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 256, 't': 8192, 'seed': 0},
    large_param_points=tuple(
        {'n_sig': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'two product_filter passes (forward + flipped) -> 2x rfft/irfft: '
        'O(N log t), GPU-pure. HBM ~ batch*t. The size tier grows batch x '
        'length.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
