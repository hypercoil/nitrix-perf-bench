# -*- coding: utf-8 -*-
"""Registration metric: ``nitrix.metrics.lncc`` vs numpy / SimpleITK / cupy.

Local (windowed) normalised cross-correlation -- the ANTs squared local-CC, the
diffeomorphic-recipe workhorse (robust to smooth intensity inhomogeneity).  Per
voxel ``cc = (Σ m̃ f̃)² / (Σ m̃² · Σ f̃²)`` over a box window (radius 4), via
separable box sums.

Warranted comparison + the convention gap (nitrix documents it; verified
2026-06-09): the numpy reimplementation (separable ``correlate1d`` box sums,
reflect boundary -- matches nitrix's box-sum boundary to fp32 round-off) is the
fp64 oracle + CPU floor; ``cupy`` is the GPU bar.  ITK's
``ANTSNeighborhoodCorrelation`` is **interior-identical** to nitrix in fp64,
but returns it as a *cost* (negated voxel mean) and trims the boundary to valid
neighbourhoods (nitrix keeps a uniform reflect count), so the scalar diverges
by sign + boundary -> it rides as a **labelled divergent** ``ApproxBaseline``
(reported, not gated).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.metrics import lncc as nx_lncc

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._metrics import _sitk_metric_eval, cupy_lncc, metric_pair, np_lncc

_RADIUS = 4


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = param['shape']
    m, f = metric_pair(shape, param.get('seed', 0), 'within')
    mj = jax.block_until_ready(jnp.asarray(m))
    fj = jax.block_until_ready(jnp.asarray(f))

    ref = np_lncc(m, f, _RADIUS)  # fp64 oracle (ANTs squared local CC)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(m, f)
        return (m, f) if framework == 'numpy' else (mj, fj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: nx_lncc(a, b)),
        'numpy.lncc': ('numpy', lambda a, b: np_lncc(a, b, _RADIUS)),
        'simpleitk.ANTSNeighborhoodCorrelation': (
            'simpleitk', _sitk_metric_eval(
                'SetMetricAsANTSNeighborhoodCorrelation', _RADIUS)),
        'cupy.lncc': ('cupy', cupy_lncc(_RADIUS)),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64, 64], [128, 128, 128]]

CASE = Case(
    name='lncc',
    op_qualname='nitrix.metrics.lncc',
    output_independent=False,  # windowed + reduced over the image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'simpleitk.ANTSNeighborhoodCorrelation',
            'ITK ANTSNeighborhoodCorrelation is interior-identical (fp64) to '
            'nitrix.lncc but returns the negated voxel-mean cost and trims to '
            'valid neighbourhoods (vs nitrix reflect), so the scalar diverges '
            'by sign + boundary -- the documented convention gap, not an '
            'error (verified on this L4, 2026-06-09).'),
    ),
)
