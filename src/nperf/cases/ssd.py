# -*- coding: utf-8 -*-
"""Registration metric: ``nitrix.metrics.ssd`` vs numpy / SimpleITK / cupy.

Mean squared difference -- the within-modality cost (motion correction), and
the default-metric of the rigid/affine recipe.  A single elementwise pass:
``mean((moving - fixed)**2)``.

Warranted comparison: nitrix's default ``reduction='mean'`` is **bit-equal in
fp64** to ITK ``MeanSquares`` (nitrix documents this; verified 2026-06-09), so
``simpleitk.MeanSquares`` is a genuine **co-oracle**, not just a floor -- the
numpy reimplementation (the fp64 oracle + CPU floor) and SimpleITK both gate
against nitrix.  cupy is the on-target GPU bar.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.metrics import ssd as nx_ssd

from ._base import BuiltPoint, Case, to_cupy
from ._metrics import _sitk_metric_eval, cupy_ssd, metric_pair, np_ssd


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = param['shape']
    m, f = metric_pair(shape, param.get('seed', 0), 'within')
    mj = jax.block_until_ready(jnp.asarray(m))
    fj = jax.block_until_ready(jnp.asarray(f))

    ref = np_ssd(m, f)  # fp64 oracle (== ITK MeanSquares)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(m, f)
        return (m, f) if framework == 'numpy' else (mj, fj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: nx_ssd(a, b)),
        'numpy.ssd': ('numpy', np_ssd),  # CPU floor + oracle
        'simpleitk.MeanSquares': (
            'simpleitk', _sitk_metric_eval('SetMetricAsMeanSquares')),
        'cupy.ssd': ('cupy', cupy_ssd()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64, 64], [128, 128, 128]]

CASE = Case(
    name='ssd',
    op_qualname='nitrix.metrics.ssd',
    output_independent=False,  # a global reduction over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
