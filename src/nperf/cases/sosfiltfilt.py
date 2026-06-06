# -*- coding: utf-8 -*-
"""Tier-2 signal-filter: ``nitrix.signal.sosfiltfilt`` vs scipy / cupy.

Zero-phase forward-backward IIR (SOS Butterworth) filtering -- the standard
fMRI / EEG band-pass (cancels phase, squares the magnitude response).  Like
``sosfilt``, the default is ``backend='auto'`` -- the **FFT-convolution engine
on GPU**, ``lax.scan`` on CPU (B18 Win 2): the zero-phase path is no longer
scan-only, the FFT engine adds the ``zi`` transient (``x[0]*g``) over the first
``n_taps`` samples so the edges stay scipy-exact.  This case measures the
default (no kwarg) as the headline ``nitrix-jax`` row, with the ``fft`` /
``scan`` engines as labelled variants.

The order-8 long-series row is the forward-backward *fidelity* guard B18 asked
for (the doubled pass + odd padding is where transient bugs hide): nitrix
matches ``scipy.signal.sosfiltfilt`` to fp32 round-off there.  scipy is the CPU
floor + fp64 oracle; cupy is the on-target GPU ref.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.signal import sosfiltfilt

from ._base import BuiltPoint, Case, to_cupy
from ._filters import cupy_sosfilt, design_sos, iir_input, scipy_sosfilt


def _build(param: Dict[str, Any]) -> BuiltPoint:
    ch, obs = param['channels'], param['obs']
    X = iir_input(ch, obs, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    sos = design_sos(order=param.get('order', 4))

    ref = scipy_sosfilt(sos.astype(np.float64), filtfilt=True)(
        X.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        # default call (no backend) -- fft on GPU, scan on CPU.
        'nitrix-jax': ('jax', lambda x: sosfiltfilt(x, sos)),
        'nitrix-jax-fft': (
            'jax', lambda x: sosfiltfilt(x, sos, backend='fft')),
        'nitrix-jax-scan': (
            'jax', lambda x: sosfiltfilt(x, sos, backend='scan')),
        'scipy.signal.sosfiltfilt': (
            'scipy', scipy_sosfilt(sos, filtfilt=True)),  # CPU floor
        'cupyx.scipy.signal.sosfiltfilt': (
            'cupy', cupy_sosfilt(sos, filtfilt=True)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, obs, order): short / multichannel moderate / multichannel + FFT
# regime / order-8 long-series (the forward-backward fidelity guard, B18).
_POINTS = [
    {'channels': 64, 'obs': 4096, 'order': 4},     # short signal
    {'channels': 256, 'obs': 8192, 'order': 4},    # multichannel, moderate
    {'channels': 256, 'obs': 32768, 'order': 4},   # multichannel + FFT regime
    {'channels': 64, 'obs': 65536, 'order': 8},    # order-8 long-series guard
]

CASE = Case(
    name='sosfiltfilt',
    op_qualname='nitrix.signal.sosfiltfilt',
    output_independent=False,  # forward-backward recurrence couples all time
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    representative={'channels': 256, 'obs': 8192, 'order': 4, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,  # fp32; FFT ~1e-6, scan ~3e-6 vs scipy fp64 (all pass)
)
