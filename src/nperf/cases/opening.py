# -*- coding: utf-8 -*-
"""Tier-2 morphology: ``nitrix.morphology.open`` vs scipy / cupy.

Morphological opening (``dilate(erode(x))``) -- removes small bright
structures.  A two-pass op: it **inherits whichever branch its erode/dilate
hit**, so the flat-box path is doubly-fast (two fused ``reduce_window`` passes)
and an explicit disk/ball footprint is doubly-slow (two ``semiring_conv``
passes).  This case measures the box (fast) and disk/ball (slow) points so the
compounding is visible (B18 Win 3).

The scipy / cupy reference composes ``grey_erosion`` then ``grey_dilation``
explicitly, each with its own matching constant-pad cval (``+inf`` then
``-inf``) -- scipy's single-``cval`` ``grey_opening`` cannot match nitrix's two
different border identities, so we compose to compare the *same* op (B13).  See
``_morphology.py``.  Ratio vs ``nitrix-jax``; fp64 oracle = the composed scipy.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.morphology import open as morph_open

from ._base import BuiltPoint, Case, to_cupy
from ._morphology import (
    cupy_morph,
    morph_input,
    nitrix_kwargs,
    resolve_se,
    scipy_morph,
)

_KIND = 'open'


def _build(param: Dict[str, Any]) -> BuiltPoint:
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
        'nitrix-jax': ('jax', lambda x: morph_open(x, **kw)),
        'scipy.ndimage.grey_opening': (
            'scipy', scipy_morph(_KIND, se_spec)),  # CPU floor + oracle
        'cupyx.scipy.ndimage.grey_opening': (
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

CASE = Case(
    name='open',
    op_qualname='nitrix.morphology.open',
    output_independent=False,  # erode+dilate couple a structuring-element nbhd
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    representative={'shape': [256, 256], 'se': 'box', 'size': 3, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
