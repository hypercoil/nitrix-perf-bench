# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.random_resized_crop`` vs scipy / cupy.

Sample a random sub-window (extent ~ U(scale_range)·shape), resample it to a
fixed ``size`` (trilinear) -- the 3-D DINOv2 RandomResizedCrop. Real resampling
work. **RNG op -- no cross-framework oracle** (`fp64_reference=None`): the
window is key-determined, so the ratio is a task-level wall-clock comparison
and the *structural* property (output shape == ``size`` + channel) is checked
in tests. cupy (`cupyx.scipy.ndimage.zoom`) is the GPU headline ref; numpy
(`scipy.ndimage.zoom`) the CPU floor. No MONAI row -- `RandZoom` keeps the
input shape (a zoom, not a crop-then-resize), so it is not the same op. Ratio
vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import random_resized_crop

from ._augment import augment_input, cupy_resized_crop, np_resized_crop
from ._base import BuiltPoint, Case, to_cupy

_SCALE = (0.5, 1.0)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    seed = int(param.get('seed', 0))
    shape = param['shape']
    size = param.get('size') or [s // 2 for s in shape]
    size = [int(s) for s in size]
    X = augment_input(shape, seed)[..., None]  # channels-last (single channel)
    jx = jax.block_until_ready(jnp.asarray(X))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        if framework == 'jax':
            return (jx, key)
        return (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x, k: random_resized_crop(
            x, k, size=size, scale_range=_SCALE)),
        'numpy.random_resized_crop': (
            'numpy', np_resized_crop(size, _SCALE, seed)),  # scipy zoom
        'cupy.random_resized_crop': (
            'cupy', cupy_resized_crop(size, _SCALE, seed)),  # GPU headline
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op (key-determined crop window): no cross-frame '
                      'oracle; ratio is task-level wall-clock, the output '
                      'shape == size is checked in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): crop+trilinear-resample to half size along each spatial axis.
_SIZES = [64, 96, 128]

CASE = Case(
    name='random_resized_crop',
    op_qualname='nitrix.augment.random_resized_crop',
    output_independent=False,  # each output samples a window-dependent stencil
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'seed': 0} for s in _SIZES],
    representative={'shape': [96, 96, 96], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
