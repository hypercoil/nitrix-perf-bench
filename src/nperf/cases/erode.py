# -*- coding: utf-8 -*-
"""Tier-2 morphology: ``nitrix.morphology.erode`` vs scipy / cupy / ITK.

Grey erosion (``out[i] = min_p(x[i+p] - se[p])``), the min-plus sibling of the
``dilate`` case.  Hardened identically for B18 Win 3: the flat-box default
lowers to a fused ``lax.reduce_window`` (the **fast path**), but **any explicit
structuring element -- including a flat disk / ball footprint -- routes through
the slow ``semiring_conv`` (im2col + tropical-matmul) path.**  The case
measures both SE shapes (box + disk/ball) as param points so the slow-disk gap
(nitrix vs cupy native) is visible rather than hidden by a box-only bench.

Border pinned to nitrix's SAME + ``+inf`` identity via the scipy/cupy oracle's
``mode='constant', cval=+inf`` (the *same* op, not scipy's default reflect;
B13).  See ``_morphology.py`` for the SE-encoding contract.  Ratio vs
``nitrix-jax``; fp64 oracle = scipy ``grey_erosion``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.morphology import erode

from ._base import BuiltPoint, Case, to_cupy
from ._itk import sitk_grey_morph
from ._morphology import (
    build_morph_large,
    cupy_morph,
    morph_input,
    nitrix_kwargs,
    resolve_se,
    scipy_morph,
)

_KIND = 'erode'


def _build(param: Dict[str, Any]) -> BuiltPoint:
    if param.get('tier') == 'large':  # brain-scale size tier (nitrix + cupy)
        return build_morph_large(_KIND, erode, param)
    dtype = param.get('dtype', 'float32')
    se_spec, se = resolve_se(param, dtype)
    X = morph_input(param['shape'], param.get('seed', 0), dtype)
    jx = jax.block_until_ready(jnp.asarray(X))
    se_jax = None if se is None else jnp.asarray(se)
    kw = nitrix_kwargs(se_spec, se_jax)

    ref = scipy_morph(_KIND, se_spec)(X.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {'nitrix-jax': ('jax', lambda x: erode(x, **kw))}
    if dtype == 'float32':
        # scipy / cupy have no native fp16 morphology path, so the fp16 row is
        # nitrix-only -- a half-precision perf/memory probe against the fp64
        # oracle (min/max is precision-robust, so it stays exact).
        baselines['scipy.ndimage.grey_erosion'] = (
            'scipy', scipy_morph(_KIND, se_spec))  # CPU floor + oracle
        baselines['cupyx.scipy.ndimage.grey_erosion'] = (
            'cupy', cupy_morph(_KIND, se_spec))  # GPU on-target ref
        if se_spec['kind'] == 'box':
            sz = kw['size']
            baselines['simpleitk.GrayscaleErode'] = (
                'simpleitk', lambda x: sitk_grey_morph('erode')(x, sz))
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_POINTS = [
    {'shape': [256, 256], 'se': 'box', 'size': 3},      # fast path 2-D (rep)
    {'shape': [256, 256], 'se': 'box', 'size': 15},     # large window, fast
    {'shape': [256, 256], 'se': 'disk', 'radius': 3},   # slow path 2-D (gap)
    {'shape': [256, 256], 'se': 'disk', 'radius': 7},   # large disk, slow
    {'shape': [64, 64, 64], 'se': 'box', 'size': 3},    # 3-D volumetric, fast
    {'shape': [64, 64, 64], 'se': 'ball', 'radius': 2},  # 3-D ball, slow
    {'shape': [256, 256], 'se': 'box', 'size': 3,
     'dtype': 'float16'},                               # fp16 precision row
]

# Brain-scale size tier (scale-gaming defence, COVERAGE_MANDATE §2.6): mirrors
# dilate -- flat-box fast path vs the disk/ball im2col that OOMs at 256^3, plus
# a cohort batch.  nitrix + cupy only (scale, not fidelity).
_LARGE = [
    {'shape': [256, 256, 256], 'se': 'box', 'size': 3},     # fast box
    {'shape': [256, 256, 256], 'se': 'ball', 'radius': 2},  # im2col hog
    {'shape': [256, 256, 256], 'se': 'ball', 'radius': 4},  # im2col OOM
    {'shape': [128, 128, 128], 'se': 'ball', 'radius': 2,
     'batch': 4},                                           # cohort hog
]

CASE = Case(
    name='erode',
    op_qualname='nitrix.morphology.erode',
    output_independent=False,  # output = min over a structuring element
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    large_param_points=tuple(
        {**p, 'tier': 'large', 'seed': 0} for p in _LARGE),
    representative={'shape': [256, 256], 'se': 'box', 'size': 3, 'seed': 0},
    complexity=(
        'time: flat box O(N) (fused reduce_window) vs explicit SE O(N*k^d) '
        '(im2col); HBM: box O(N), explicit-SE im2col O(N*k^d) -> 256^3 ball '
        'OOMs (~49 GB) while cupy/scipy (O(N*k), in-place) hold. The flat box '
        'scales; the disk/ball footprint does not.'
    ),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
