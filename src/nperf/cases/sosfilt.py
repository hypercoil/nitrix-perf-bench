# -*- coding: utf-8 -*-
"""Tier-2 signal-filter: ``nitrix.signal.sosfilt`` vs scipy / cupy.

Causal IIR (second-order-sections) Butterworth filtering.  nitrix's default is
``backend='auto'`` -- the **FFT-convolution engine on GPU**, the sequential
``lax.scan`` recurrence on CPU (B18 Win 2): an IIR filter is LTI, so its output
is exactly convolution with the (truncated) impulse response, which the GPU
runs latency-free.  This case measures the op **the way users call it** (no
kwarg -> the platform default) as the headline ``nitrix-jax`` row, with the
three engines (``fft`` / ``scan`` / ``associative``) as labelled variants
beside it -- so the GPU FFT win the old scan-pinned case never measured is
visible, and the engine-vs-engine comparison stays explicit.

Sizes span the regimes that decide the ``auto`` switch (B18 Win 2): a short
signal (the recurrence is competitive), long signals ``obs >= 32768`` (where
the FFT wins on GPU), a multichannel row (64-306 ch is real EEG/MEG, nitrix
vectorises), and an order-8 row (a longer impulse -> bigger FFT).  The FFT
engine is in fact *more* accurate than the recurrence in fp32 (the recurrence
accumulates error over T steps; the convolution does not), so the fp64 scipy
oracle gates all engines tightly.  scipy is the CPU floor + oracle; cupy is the
on-target GPU ref.  Ratio vs ``nitrix-jax`` (the auto default).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.signal import sosfilt

from ._base import BuiltPoint, Case, to_cupy
from ._filters import cupy_sosfilt, design_sos, iir_input, scipy_sosfilt


def _build(param: Dict[str, Any]) -> BuiltPoint:
    ch, obs = param['channels'], param['obs']
    X = iir_input(ch, obs, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    sos = design_sos(order=param.get('order', 4))  # shared by all baselines

    ref = scipy_sosfilt(sos.astype(np.float64))(X.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        # default call (no backend) -- the platform engine users actually hit
        # (fft on GPU, scan on CPU).
        'nitrix-jax': ('jax', lambda x: sosfilt(x, sos)),
        'nitrix-jax-fft': ('jax', lambda x: sosfilt(x, sos, backend='fft')),
        'nitrix-jax-scan': ('jax', lambda x: sosfilt(x, sos, backend='scan')),
        'nitrix-jax-assoc': (
            'jax', lambda x: sosfilt(x, sos, backend='associative')),
        'scipy.signal.sosfilt': ('scipy', scipy_sosfilt(sos)),  # CPU floor
        'cupyx.scipy.signal.sosfilt': ('cupy', cupy_sosfilt(sos)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, obs, order): short (recurrence competitive) / long obs>=32768 (FFT
# wins on GPU) / multichannel (EEG/MEG) / order-8 (longer impulse).
_POINTS = [
    {'channels': 64, 'obs': 4096, 'order': 4},     # short signal
    {'channels': 256, 'obs': 8192, 'order': 4},    # multichannel, moderate
    {'channels': 256, 'obs': 32768, 'order': 4},   # multichannel + FFT regime
    {'channels': 64, 'obs': 65536, 'order': 8},    # long + order-8 impulse
]

CASE = Case(
    name='sosfilt',
    op_qualname='nitrix.signal.sosfilt',
    output_independent=False,  # the recurrence couples the whole time axis
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    representative={'channels': 256, 'obs': 8192, 'order': 4, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,  # fp32; FFT ~5e-7, scan ~2.5e-6, associative ~1.1e-5 (all pass)
)
