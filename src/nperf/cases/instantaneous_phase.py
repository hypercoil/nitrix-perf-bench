# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.instantaneous_phase`` vs scipy / cupy.

Unwrapped instantaneous phase ``unwrap(angle(analytic_signal(X)))`` of a batch
of real time series -- the phase track for phase-amplitude coupling / phase
synchrony.  fp64 oracle + community baseline: scipy
(``unwrap(angle(scipy.signal.hilbert))``); GPU bar: a cupy reimpl (FFT-based,
GPU-pure).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import instantaneous_phase

from ._base import BuiltPoint, Case, to_cupy
from ._signal import cupy_inst, scipy_inst_phase, signal_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = int(param['n_sig']), int(param['t'])
    X = signal_input(n_sig, t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = scipy_inst_phase(fp64=True)(X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: instantaneous_phase(x)),
        'scipy.instantaneous_phase': ('scipy', scipy_inst_phase()),
        'cupy.instantaneous_phase': ('cupy', cupy_inst('phase')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(64, 4096), (256, 8192), (1024, 8192)]
_LARGE = [(2048, 16384), (4096, 16384)]

CASE = Case(
    name='instantaneous_phase',
    op_qualname='nitrix.stats.instantaneous_phase',
    output_independent=False,  # FFT + a sequential unwrap along the time axis
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 256, 't': 8192, 'seed': 0},
    large_param_points=tuple(
        {'n_sig': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'an FFT Hilbert (O(N log t)) + angle + a cumulative unwrap along the '
        'time axis; GPU-pure (no cuSolver), a clean cupy bar. '
        'HBM ~ batch*t (the complex analytic signal). The size tier grows the '
        'batch x length.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
