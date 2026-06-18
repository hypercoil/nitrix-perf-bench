# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.env_inst`` vs scipy / cupy.

The FUSED analytic-signal readout: ``(envelope, instantaneous_frequency,
instantaneous_phase)`` from ONE ``analytic_signal`` (Hilbert) call -- the perf
win is sharing the single FFT across all three derived tracks (vs three
separate calls).  To force the full fused compute (no XLA dead-code elimination
of unused outputs) and give a clean scalar fidelity, the baseline reduces the
three outputs to their summed scalar.  fp64 oracle + community baseline: scipy
(one ``hilbert`` -> abs / unwrap-angle / diff); GPU bar: a cupy reimpl.  Ratio
vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import env_inst

from ._base import BuiltPoint, Case, to_cupy
from ._signal import cupy_env_inst_sum, scipy_env_inst_sum, signal_input

_PERIOD = 2.0 * 3.141592653589793


def _nitrix_sum(x: Any) -> Any:
    env, freq, phase = env_inst(x)        # one analytic_signal, three readouts
    return env.sum() + freq.sum() + phase.sum()


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_sig, t = int(param['n_sig']), int(param['t'])
    X = signal_input(n_sig, t, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = scipy_env_inst_sum(fp64=True)(X)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (jx,) if framework == 'jax' else (X,)

    baselines = {
        'nitrix-jax': ('jax', _nitrix_sum),
        'scipy.env_inst': ('scipy', scipy_env_inst_sum()),
        'cupy.env_inst': ('cupy', cupy_env_inst_sum()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(64, 4096), (256, 8192), (1024, 8192)]
_LARGE = [(2048, 16384), (4096, 16384)]

CASE = Case(
    name='env_inst',
    op_qualname='nitrix.stats.env_inst',
    output_independent=False,  # one shared FFT feeding three reductions
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n_sig': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n_sig': 256, 't': 8192, 'seed': 0},
    large_param_points=tuple(
        {'n_sig': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'ONE FFT Hilbert shared across envelope + inst-freq + inst-phase '
        '(vs three separate analytic_signal calls -- the fusion is the win); '
        'GPU-pure. HBM ~ batch*t. Read as a summed scalar to force all three '
        'outputs. The size tier grows the batch x length.'),
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
