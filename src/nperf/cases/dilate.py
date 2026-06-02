# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.morphology.dilate`` vs scipy.ndimage.

Grey dilation (windowed max), sibling of the ``erode`` case: nitrix on jax vs
``scipy.ndimage.grey_dilation`` (host; the ``scipy`` provider).  Exact in any
precision (a windowed max picks an input value), so both agree with the fp64
oracle.  Ratio via ``--reference scipy.ndimage.grey_dilation``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.morphology import dilate

from ._base import BuiltPoint, Case, to_cupy
from ._itk import sitk_grey_morph


def _cupy_dilate(x: Any, size: int) -> Any:
    '''GPU grey_dilation (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).'''
    from cupyx.scipy import ndimage as cnd

    return cnd.grey_dilation(x, size=size)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    size = param.get('size', 3)
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal(shape).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = spnd.grey_dilation(X.astype(np.float64), size=size)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: dilate(x, size=size)),
        'scipy.ndimage.grey_dilation': (
            'scipy', lambda x: spnd.grey_dilation(x, size=size)),
        'simpleitk.GrayscaleDilate': (  # ITK floor (exact match, verified)
            'simpleitk', lambda x: sitk_grey_morph('dilate')(x, size)),
        'cupyx.scipy.ndimage.grey_dilation': (
            'cupy', lambda x: _cupy_dilate(x, size)),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64], [256, 256]]

CASE = Case(
    name='dilate',
    op_qualname='nitrix.morphology.dilate',
    output_independent=False,  # each output is a max over a size-window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'size': 3, 'seed': 0} for s in _SHAPES],
    representative={'shape': [256, 256], 'size': 3, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
