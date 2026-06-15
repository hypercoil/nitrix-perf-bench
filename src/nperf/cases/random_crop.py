# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.random_crop`` vs MONAI / cupy.

Random-offset fixed-size crop (`lax.dynamic_slice`). Memory-bound (a strided
copy of the sub-block). **RNG op -- no cross-framework oracle**
(`fp64_reference=None`): the offset is key-determined, so the ratio is a
task-level wall-clock comparison and a *structural* property (output shape ==
``size`` and values lie within the input range -- a contiguous sub-block) is
checked in tests. cupy is the GPU headline ref; numpy the CPU floor; MONAI
`RandSpatialCrop` the community baseline (timing). Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import random_crop

from ._augment import (
    augment_input,
    cupy_random_crop,
    monai_random_transform,
    np_random_crop,
)
from ._base import BuiltPoint, Case, to_cupy


def _build(param: Dict[str, Any]) -> BuiltPoint:
    seed = int(param.get('seed', 0))
    shape = param['shape']
    size = param.get('size') or [s // 2 for s in shape]
    size = [int(s) for s in size]
    X = augment_input(shape, seed)
    jx = jax.block_until_ready(jnp.asarray(X))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        if framework == 'jax':
            return (jx, key)
        return (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x, k: random_crop(x, k, size=size)),
        'numpy.random_crop': ('numpy', np_random_crop(size, seed)),
        'cupy.random_crop': ('cupy', cupy_random_crop(size, seed)),
        'monai.RandSpatialCrop': (
            'monai', monai_random_transform('crop', size=size)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op (key-determined offset): no cross-framework '
                      'oracle; ratio is task-level wall-clock, the '
                      'sub-block (shape + value-range) property is in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): crop to half size along each axis.
_SIZES = [64, 96, 128]

CASE = Case(
    name='random_crop',
    op_qualname='nitrix.augment.random_crop',
    output_independent=True,  # each output voxel is a copied input voxel
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'seed': 0} for s in _SIZES],
    representative={'shape': [96, 96, 96], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
