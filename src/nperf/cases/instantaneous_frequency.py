# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.signal.instantaneous_frequency`` vs scipy / cupy.

Instantaneous frequency ``fs * diff(unwrap(angle(analytic))) / period`` of a
batch of real time series (the output time axis is one shorter -- a discrete
derivative).  fp64 oracle + community baseline: scipy (hilbert -> angle ->
unwrap -> diff); GPU bar: a cupy reimpl (FFT-based, GPU-pure).  Ratio vs
``nitrix-jax``.

**Input is a narrowband chirp, NOT white noise** (``narrowband_signal``):
instantaneous frequency is only well-posed on a narrowband signal.  On
broadband input it is unstable at Nyquist (the +-pi wrap ambiguity) and where
the envelope nears zero -- by NATURE, not fp32 or implementation (the
conjugate-product reformulation was verified not to help).  See nitrix FR
``doc-instantaneous-frequency-narrowband-caveat``.  On the chirp (the textbook
regime) fp32 matches the fp64 oracle.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.signal import instantaneous_frequency

from ._base import BuiltPoint, Case, to_cupy
from ._signal import cupy_inst, narrowband_signal, scipy_inst_freq


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = int(param['n_sig']), int(param['t'])
    X = narrowband_signal(n_sig, t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = scipy_inst_freq(fp64=True)(X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: instantaneous_frequency(x)),
        'scipy.instantaneous_frequency': ('scipy', scipy_inst_freq()),
        'cupy.instantaneous_frequency': ('cupy', cupy_inst('freq')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(64, 4096), (256, 8192), (1024, 8192)]
_LARGE = [(2048, 16384), (4096, 16384)]

CASE = Case(
    name='instantaneous_frequency',
    op_qualname='nitrix.signal.instantaneous_frequency',
    output_independent=False,  # FFT + sequential unwrap, then a diff
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 256, 't': 8192, 'seed': 0},
    large_param_points=tuple(
        {'n_sig': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'instantaneous_phase (FFT Hilbert + unwrap) + a trailing diff; '
        'GPU-pure, a clean cupy bar. HBM ~ batch*t. The size tier grows the '
        'batch x length.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
