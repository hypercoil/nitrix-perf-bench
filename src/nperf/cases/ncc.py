# -*- coding: utf-8 -*-
"""Registration metric: ``nitrix.metrics.ncc`` vs numpy / SimpleITK / cupy.

Global normalised cross-correlation -- the *signed Pearson correlation* over
all voxels (within-modality, robust to a linear intensity change).  A two-pass
reduction (means, then the covariance / variances).

Warranted comparison + the convention gap (nitrix documents it; verified
2026-06-09): nitrix ``ncc`` returns the **signed** Pearson ``r``; the numpy
Pearson reimplementation is the fp64 oracle + CPU floor and ``cupy`` the GPU
bar -- both gate against nitrix.  ITK's ``Correlation`` metric, by contrast,
returns ``-r**2`` (squared -> sign-dropped, negated for minimisation): a
**different quantity**, recoverable only one-way as ``-ncc**2``.  So
``simpleitk.Correlation`` rides as a **labelled divergent** ``ApproxBaseline``
(its speed is comparable-class, its "fidelity" vs the ``r`` oracle is the
*documented convention gap*, reported not gated) -- it is not an error, and not
an oracle.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.metrics import ncc as nx_ncc

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._metrics import _sitk_metric_eval, cupy_ncc, metric_pair, np_ncc


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = param['shape']
    m, f = metric_pair(shape, param.get('seed', 0), 'within')
    mj = jax.block_until_ready(jnp.asarray(m))
    fj = jax.block_until_ready(jnp.asarray(f))

    ref = np_ncc(m, f)  # fp64 oracle (signed Pearson r)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(m, f)
        return (m, f) if framework == 'numpy' else (mj, fj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: nx_ncc(a, b)),
        'numpy.ncc': ('numpy', np_ncc),  # CPU floor + oracle
        'simpleitk.Correlation': (
            'simpleitk', _sitk_metric_eval('SetMetricAsCorrelation')),
        'cupy.ncc': ('cupy', cupy_ncc()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64, 64], [128, 128, 128]]

CASE = Case(
    name='ncc',
    op_qualname='nitrix.metrics.ncc',
    output_independent=False,  # a global reduction over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'simpleitk.Correlation',
            'ITK CorrelationImageToImageMetricv4 returns -r**2 (squared, '
            'sign-dropped, negated for minimisation), not the signed Pearson '
            'r that nitrix.ncc returns -- recover one-way as -ncc**2. The '
            'fidelity vs the r oracle is the documented convention gap, not '
            'an error (verified on this L4, 2026-06-09).'),
    ),
)
