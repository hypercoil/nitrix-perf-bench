# -*- coding: utf-8 -*-
"""Tier-2 signal-filter: ``nitrix.signal.sosfilt`` vs scipy / cupy.

Causal IIR (second-order-sections) Butterworth filtering -- the first
**recursive** op in the suite (each output depends on previous outputs). nitrix
exposes two recurrence engines, both benchmarked here as separate baselines:
``backend='scan'`` (sequential ``lax.scan``, O(T) depth) and
``backend='associative'`` (parallel-prefix ``lax.associative_scan``,
O(log T) depth). The headline is sequential-vs-parallel on the GPU: the scan is
~70x slower than the associative engine on the L4, where the recurrence depth
dominates.

scipy.signal.sosfilt is the CPU floor + fp64 oracle; ``cupyx.scipy.signal.
sosfilt`` is the on-target GPU reference. Same SOS coefficients feed all four
(see ``cases/_filters.py``). Ratio vs ``nitrix-jax`` (the default scan backend),
so the associative row advertises its speedup over the default-API path.
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
    sos = design_sos()  # order-4 band-pass, shared by all baselines

    ref = scipy_sosfilt(sos.astype(np.float64))(X.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        # the default-API call (backend='scan') is the canonical SUT row;
        # the parallel-prefix engine is the labelled variant beside it.
        'nitrix-jax': ('jax', lambda x: sosfilt(x, sos, backend='scan')),
        'nitrix-jax-assoc': (
            'jax', lambda x: sosfilt(x, sos, backend='associative')),
        'scipy.signal.sosfilt': ('scipy', scipy_sosfilt(sos)),  # CPU floor
        'cupyx.scipy.signal.sosfilt': ('cupy', cupy_sosfilt(sos)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, observations): cost ~ channels * obs * n_sections; the scan's
# O(obs) recurrence depth is what hurts on the GPU as obs grows.
_SHAPES = [(512, 2048), (1024, 4096), (2048, 8192)]

CASE = Case(
    name='sosfilt',
    op_qualname='nitrix.signal.sosfilt',
    output_independent=False,  # the recurrence couples the whole time axis
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'channels': c, 'obs': o, 'seed': 0} for (c, o) in _SHAPES],
    representative={'channels': 1024, 'obs': 4096, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,  # fp32 recurrence; the associative engine accumulates ~2e-5
)
