# -*- coding: utf-8 -*-
"""Tier-2 signal-filter: ``nitrix.signal.sosfiltfilt`` vs scipy / cupy.

Zero-phase forward-backward IIR (SOS Butterworth) filtering -- the standard
fMRI band-pass (cancels phase, squares the magnitude response). Recursive and
sequential (the zero-phase path is ``lax.scan``-only -- forward then backward
with scipy-exact steady-state initial conditions + odd padding), so like
``sosfilt`` its recurrence depth is O(T).

scipy.signal.sosfiltfilt is the CPU floor + fp64 oracle; ``cupyx.scipy.signal.
sosfiltfilt`` is the on-target GPU reference. Same SOS coefficients feed all
three (see ``cases/_filters.py``). Ratio vs ``nitrix-jax``.
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
    sos = design_sos()

    ref = scipy_sosfilt(sos.astype(np.float64), filtfilt=True)(
        X.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: sosfiltfilt(x, sos)),
        'scipy.signal.sosfiltfilt': (
            'scipy', scipy_sosfilt(sos, filtfilt=True)),  # CPU floor
        'cupyx.scipy.signal.sosfiltfilt': (
            'cupy', cupy_sosfilt(sos, filtfilt=True)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, observations): forward+backward, so ~2x sosfilt's sequential cost.
_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192)]

CASE = Case(
    name='sosfiltfilt',
    op_qualname='nitrix.signal.sosfiltfilt',
    output_independent=False,  # forward-backward recurrence couples all time
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'channels': c, 'obs': o, 'seed': 0} for (c, o) in _SHAPES],
    representative={'channels': 1024, 'obs': 4096, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
