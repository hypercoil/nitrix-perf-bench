# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.signal.hilbert_transform`` vs scipy / cupy.

The Hilbert transform (imaginary part of the analytic signal), sibling of the
``analytic_signal`` case (see it + ``cases/_signal.py``).  Real-valued output.
nitrix (jax) vs ``scipy.signal.hilbert(.).imag`` (CPU floor) +
``cupyx.scipy.signal.hilbert(.).imag`` (GPU ref), fp64 scipy oracle.  FFT-based
and GPU-pure.  Ratio vs ``scipy.signal.hilbert``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as ss
from nitrix.signal import hilbert_transform

from ._base import BuiltPoint, Case, to_cupy
from ._signal import cupy_hilbert, signal_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = param['n_sig'], param['t']
    X = signal_input(n_sig, t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = ss.hilbert(X.astype(np.float64), axis=-1).imag  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: hilbert_transform(x)),
        'scipy.signal.hilbert': (
            'scipy', lambda x: ss.hilbert(x, axis=-1).imag),
        'cupyx.scipy.signal.hilbert': ('cupy', cupy_hilbert('imag')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(512, 1024), (2048, 2048), (4096, 4096)]

CASE = Case(
    name='hilbert_transform',
    op_qualname='nitrix.signal.hilbert_transform',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 2048, 't': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
