# -*- coding: utf-8 -*-
"""Registration metric: ``nitrix.metrics.correlation_ratio`` vs numpy / cupy.

Roche's correlation ratio ``η²(moving | fixed)`` -- the FSL ``mcflirt`` cost
family, which assumes only a *functional* (not affine) intensity relationship
(cross-modal).  Soft-bins ``fixed`` into groups (bins=32) and measures how much
of ``moving``'s variance the (soft) group means explain.

Warranted comparison (nitrix documents it; verified 2026-06-09): the numpy
reimplementation of the *same* soft-binned variance ratio is the fp64 oracle +
CPU floor, and ``cupy`` (soft grouping via ``bincount``) is the GPU bar -- both
gate against nitrix.  **No domain co-oracle exists**: SimpleITK's registration
framework ships no correlation-ratio metric (FSL/Roche lineage only), so unlike
the other metrics there is no divergent domain reference to label here.  Ratio
vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.metrics import correlation_ratio as nx_cr

from ._base import BuiltPoint, Case, to_cupy
from ._metrics import cupy_cr, metric_pair, np_cr

_BINS = 32


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = param['shape']
    m, f = metric_pair(shape, param.get('seed', 0), 'cross')  # cross-modal
    mj = jax.block_until_ready(jnp.asarray(m))
    fj = jax.block_until_ready(jnp.asarray(f))

    ref = np_cr(m, f, _BINS)  # fp64 oracle (Roche eta^2)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(m, f)
        return (m, f) if framework == 'numpy' else (mj, fj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: nx_cr(a, b)),
        'numpy.cr': ('numpy', lambda a, b: np_cr(a, b, _BINS)),
        'cupy.cr': ('cupy', cupy_cr(_BINS)),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64, 64], [128, 128, 128]]

CASE = Case(
    name='correlation_ratio',
    op_qualname='nitrix.metrics.correlation_ratio',
    output_independent=False,  # a global reduction (soft grouping + variance)
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
