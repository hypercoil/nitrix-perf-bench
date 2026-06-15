# -*- coding: utf-8 -*-
"""Tier-2 augment: ``nitrix.augment.random_histogram_shift`` vs MONAI / cupy.

Random monotone piecewise-linear intensity remap: ``n_control_points`` equally
spaced reference levels are perturbed (endpoints pinned, cumulative-max for
monotonicity) and applied via ``interp``. **RNG op -- no cross-framework
oracle** (`fp64_reference=None`): the control-point offsets are key-determined,
so the ratio is a task-level wall-clock comparison and the defining *property*
(the remap is monotone -> it preserves the rank order of the voxels) is checked
in tests. cupy is the GPU headline ref; numpy the CPU floor; MONAI
`RandHistogramShift` the community baseline. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import random_histogram_shift

from ._augment import (
    augment_input,
    cupy_random_histogram_shift,
    monai_random_transform,
    np_random_histogram_shift,
)
from ._base import BuiltPoint, Case, to_cupy

_N_CP = 10
_SHIFT = (-0.1, 0.1)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n_cp = int(param.get('n_control_points', _N_CP))
    seed = int(param.get('seed', 0))
    X = augment_input(param['shape'], seed)
    jx = jax.block_until_ready(jnp.asarray(X))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        if framework == 'jax':
            return (jx, key)
        return (X,)

    baselines = {
        'nitrix-jax': ('jax', lambda x, k: random_histogram_shift(
            x, k, n_control_points=n_cp, shift_range=_SHIFT)),
        'numpy.random_histogram_shift': (
            'numpy', np_random_histogram_shift(n_cp, _SHIFT, seed)),
        'cupy.random_histogram_shift': (
            'cupy', cupy_random_histogram_shift(n_cp, _SHIFT, seed)),
        'monai.RandHistogramShift': (
            'monai', monai_random_transform('hist', n_cp=n_cp)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op (key-determined control points): no cross-'
                      'framework oracle; ratio is task-level wall-clock, the '
                      'monotone (rank-preserving) property is in tests.',
        ratio_reference='nitrix-jax',
    )


_SIZES = [64, 96, 128]

CASE = Case(
    name='random_histogram_shift',
    op_qualname='nitrix.augment.random_histogram_shift',
    output_independent=False,  # the remap table couples via global min/max
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'n_control_points': _N_CP, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'n_control_points': _N_CP,
                    'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
