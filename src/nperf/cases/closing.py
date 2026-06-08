# -*- coding: utf-8 -*-
"""Tier-2 morphology: ``nitrix.morphology.close`` vs scipy / cupy.

Morphological closing (``erode(dilate(x))``) -- fills small dark holes.  The
min-plus sibling of the ``open`` case and, like it, a two-pass op that
**inherits whichever branch its dilate/erode hit**: doubly-fast on the flat
box, doubly-slow on an explicit disk/ball footprint.  Box (fast) and disk/ball
(slow) points make the compounding visible (B18 Win 3).

The scipy / cupy reference composes ``grey_dilation`` then ``grey_erosion``
explicitly, each with its own matching constant-pad cval (``-inf`` then
``+inf``), so it is the *same* op nitrix computes -- not scipy's
single-``cval`` ``grey_closing`` (B13).  See ``_morphology.py``.  Ratio vs
``nitrix-jax``; fp64 oracle = the composed scipy.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.morphology import close as morph_close

from ._base import BuiltPoint, Case, to_cupy
from ._morphology import (
    build_morph_large,
    cupy_morph,
    morph_input,
    nitrix_kwargs,
    resolve_se,
    scipy_morph,
)

_KIND = 'close'


def _build(param: Dict[str, Any]) -> BuiltPoint:
    if param.get('tier') == 'large':  # brain-scale size tier (nitrix + cupy)
        return build_morph_large(_KIND, morph_close, param)
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

    baselines = {
        'nitrix-jax': ('jax', lambda x: morph_close(x, **kw)),
        'scipy.ndimage.grey_closing': (
            'scipy', scipy_morph(_KIND, se_spec)),  # CPU floor + oracle
        'cupyx.scipy.ndimage.grey_closing': (
            'cupy', cupy_morph(_KIND, se_spec)),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_POINTS = [
    {'shape': [256, 256], 'se': 'box', 'size': 3},      # fast path 2-D (rep)
    {'shape': [256, 256], 'se': 'disk', 'radius': 3},   # slow path 2-D (gap)
    {'shape': [64, 64, 64], 'se': 'box', 'size': 3},    # 3-D volumetric, fast
    {'shape': [64, 64, 64], 'se': 'ball', 'radius': 2},  # 3-D ball, slow
]

# Brain-scale size tier (COVERAGE_MANDATE §2.6): two-pass op; flat box
# doubly-fast, disk/ball doubly the im2col, which OOMs at 256^3. nitrix + cupy.
_LARGE = [
    {'shape': [256, 256, 256], 'se': 'box', 'size': 3},     # fast box
    {'shape': [256, 256, 256], 'se': 'ball', 'radius': 2},  # im2col hog
    {'shape': [256, 256, 256], 'se': 'ball', 'radius': 4},  # im2col OOM
    {'shape': [128, 128, 128], 'se': 'ball', 'radius': 2,
     'batch': 4},                                           # cohort hog
]

CASE = Case(
    name='close',
    op_qualname='nitrix.morphology.close',
    output_independent=False,  # dilate+erode couple a structuring-element nbhd
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    large_param_points=tuple(
        {**p, 'tier': 'large', 'seed': 0} for p in _LARGE),
    representative={'shape': [256, 256], 'se': 'box', 'size': 3, 'seed': 0},
    complexity=(
        'time: flat box O(N) (two fused reduce_windows) vs explicit SE '
        'O(N*k^d) (two im2col passes); HBM: box O(N), explicit-SE O(N*k^d) -> '
        '256^3 ball OOMs (~49 GB) while cupy (O(N*k), in-place) holds. The '
        'flat box scales; the disk/ball footprint does not.'
    ),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
