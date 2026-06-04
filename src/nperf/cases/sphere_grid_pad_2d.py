# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.sphere_grid_pad_2d`` vs numpy / cupy.

Pad a 2D equirectangular sphere image with its topology -- circular wrap on the
longitudinal axis, pole-flip-and-roll-by-W/2 on the latitudinal axis -- nitrix
(jax) vs a numpy reimplementation of the exact wrap/pole-flip (CPU floor + fp64
oracle) + a CuPy GPU ref. A nitrix-specific topology (no external library
expresses the pole flip), verified equal to 0.0 in fp64. Pure gather/roll/
concat, so GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import sphere_grid_pad_2d

from ._base import BuiltPoint, Case, to_cupy
from ._sphere_grid import cupy_sphere_pad, np_sphere_pad, sphere_grid_input

_PAD = 4  # a typical conv/morphology kernel radius


def _build(param: Dict[str, Any]) -> BuiltPoint:
    h = param['h']
    IMG = sphere_grid_input(h, param.get('seed', 0))  # (H, 2H)
    jx = jax.block_until_ready(jnp.asarray(IMG))
    ref = np_sphere_pad(_PAD)(IMG.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(IMG)
        return (IMG,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda img: sphere_grid_pad_2d(img, _PAD)),
        'numpy.sphere_grid_pad': ('numpy', np_sphere_pad(_PAD)),  # CPU floor
        'cupy.sphere_grid_pad_2d': (
            'cupy', cupy_sphere_pad(_PAD)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (latitude rows): image is (H, 2H); pad = _PAD on each side.
_SIZES = [64, 128, 256]

CASE = Case(
    name='sphere_grid_pad_2d',
    op_qualname='nitrix.geometry.sphere_grid_pad_2d',
    output_independent=True,  # each padded cell is a copy/roll of one input
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'h': h, 'seed': 0} for h in _SIZES],
    representative={'h': 128, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
