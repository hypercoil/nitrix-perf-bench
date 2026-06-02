# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.morphology.erode`` vs scipy.ndimage.

Grey erosion (windowed min) -- nitrix on jax vs ``scipy.ndimage.grey_erosion``
(host; the ``scipy`` provider, in the base env).  A windowed min/max picks one
of the input values, so the result is exact in any precision: nitrix and scipy
agree bit-for-bit with the fp64 oracle.  Ratio via
``--reference scipy.ndimage.grey_erosion``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.morphology import erode

from ._base import BuiltPoint, Case, to_cupy
from ._itk import sitk_grey_morph


def _cupy_erode(x: Any, size: int) -> Any:
    '''GPU grey_erosion (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).'''
    from cupyx.scipy import ndimage as cnd

    return cnd.grey_erosion(x, size=size)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    size = param.get('size', 3)
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal(shape).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = spnd.grey_erosion(X.astype(np.float64), size=size)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: erode(x, size=size)),
        'scipy.ndimage.grey_erosion': (
            'scipy', lambda x: spnd.grey_erosion(x, size=size)),
        'simpleitk.GrayscaleErode': (  # ITK floor (exact match, verified)
            'simpleitk', lambda x: sitk_grey_morph('erode')(x, size)),
        'cupyx.scipy.ndimage.grey_erosion': (
            'cupy', lambda x: _cupy_erode(x, size)),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64], [256, 256]]

CASE = Case(
    name='erode',
    op_qualname='nitrix.morphology.erode',
    output_independent=False,  # each output is a min over a size-window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'size': 3, 'seed': 0} for s in _SHAPES],
    representative={'shape': [256, 256], 'size': 3, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
