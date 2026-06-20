# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.signal.analytic_signal`` vs scipy / cupy.

The analytic signal (FFT Hilbert) of a batch of real time series.  nitrix (jax)
vs ``scipy.signal.hilbert`` (CPU floor) + ``cupyx.scipy.signal.hilbert`` (GPU
ref), scored against an fp64 scipy oracle.  **Complex-valued output** -- the
fidelity compare handles it via ``|out - ref|``.  FFT-based, GPU-pure (no
cuSolver); a clean apples-to-apples GPU bar.  See ``cases/_signal.py``.  Ratio
vs ``scipy.signal.hilbert``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as ss
from nitrix.signal import analytic_signal

from ._base import BuiltPoint, Case, to_cupy
from ._signal import cupy_hilbert, signal_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = param['n_sig'], param['t']
    X = signal_input(n_sig, t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = ss.hilbert(X.astype(np.float64), axis=-1)  # complex128 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: analytic_signal(x)),
        'scipy.signal.hilbert': ('scipy', lambda x: ss.hilbert(x, axis=-1)),
        'cupyx.scipy.signal.hilbert': ('cupy', cupy_hilbert('complex')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (signals, samples): FFT cost ~ n_sig * t log t.
_SHAPES = [(512, 1024), (2048, 2048), (4096, 4096)]

CASE = Case(
    name='analytic_signal',
    op_qualname='nitrix.signal.analytic_signal',
    output_independent=False,  # each output sample depends on the whole signal
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 2048, 't': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
