# -*- coding: utf-8 -*-
"""Registration metric: ``nitrix.metrics.mutual_information`` vs refs.

The cross-modal workhorse (T1<->T2, EPI<->T1): MI from a **differentiable
order-1 (linear) Parzen** soft joint histogram (bins=32).  Differentiability is
nitrix's distinguishing capability here -- the domain MI tools are not
differentiable.

Warranted comparison + the convention gap (nitrix documents it; verified
2026-06-09): the numpy reimplementation of the *same* order-1 soft-binned MI is
the fp64 oracle + CPU floor; ``cupy`` (soft histogram via ``bincount``) is the
GPU bar -- both gate against nitrix.  The domain MI tools compute a **different
number at a fixed bin count** and ride as **labelled divergent**
``ApproxBaseline``s: ITK ``MattesMutualInformation`` is an order-3 (cubic)
Parzen MI returned as a negated cost; ``sklearn.mutual_info_score`` is order-0
(hard) binning.  Same family, all converging to the true continuous MI only in
the fine-bin limit.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.metrics import mutual_information as nx_mi

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._metrics import (
    _sitk_metric_eval,
    cupy_mi,
    metric_pair,
    np_mi,
    sklearn_mi,
)

_BINS = 32


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = param['shape']
    m, f = metric_pair(shape, param.get('seed', 0), 'cross')  # cross-modal
    mj = jax.block_until_ready(jnp.asarray(m))
    fj = jax.block_until_ready(jnp.asarray(f))

    ref = np_mi(m, f, _BINS)  # fp64 oracle (order-1 Parzen MI)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(m, f)
        return (m, f) if framework == 'numpy' else (mj, fj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: nx_mi(a, b)),
        'numpy.mi': ('numpy', lambda a, b: np_mi(a, b, _BINS)),
        'simpleitk.MattesMI': (
            'simpleitk', _sitk_metric_eval(
                'SetMetricAsMattesMutualInformation', _BINS)),
        'sklearn.mutual_info': ('sklearn', sklearn_mi(_BINS)),
        'cupy.mi': ('cupy', cupy_mi(_BINS)),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64, 64], [128, 128, 128]]

CASE = Case(
    name='mutual_information',
    op_qualname='nitrix.metrics.mutual_information',
    output_independent=False,  # a global reduction (joint histogram + MI)
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'simpleitk.MattesMI',
            'ITK MattesMutualInformation is an order-3 (cubic) Parzen '
            'MI returned as a negated cost -- a different number than nitrix '
            "order-1 Parzen at a fixed bin count (Parzen order differs); "
            'converges only in the fine-bin limit. Verified L4, 2026-06-09.'),
        ApproxBaseline(
            'sklearn.mutual_info',
            'sklearn.mutual_info_score is order-0 (hard) binning -- the '
            'textbook MI, a different number than nitrix order-1 Parzen at a '
            'fixed bin count (no soft smoothing). Verified L4, 2026-06-09.'),
    ),
)
