# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.random_flip`` vs MONAI / cupy.

Independent per-axis Bernoulli(p) reflection (N-D). Memory-bound (a strided
copy). **RNG op -- no cross-framework oracle** (`fp64_reference=None`): the
flipped axes are key-determined, so the perf ratio is a task-level wall-clock
comparison and a *structural* property (the value multiset is preserved) is
checked in tests. cupy is the GPU headline ref; numpy the CPU floor; MONAI
`RandFlip` the community baseline (timing -- not bit-identical). Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import random_flip

from ._augment import (
    augment_input,
    cupy_random_flip,
    monai_random_transform,
    np_random_flip,
)
from ._base import BuiltPoint, Case, to_cupy

_P = 0.5


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = float(param.get('p', _P))
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
        'nitrix-jax': ('jax', lambda x, k: random_flip(x, k, p=p)),
        'numpy.random_flip': ('numpy', np_random_flip(p, seed)),
        'cupy.random_flip': ('cupy', cupy_random_flip(p, seed)),
        'monai.RandFlip': ('monai', monai_random_transform('flip')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op (key-determined flipped axes): no cross-'
                      'framework oracle; ratio is task-level wall-clock, the '
                      'value-multiset-preserved property is checked in tests.',
        ratio_reference='nitrix-jax',
    )


_SIZES = [64, 96, 128]

CASE = Case(
    name='random_flip',
    op_qualname='nitrix.augment.random_flip',
    output_independent=False,  # a reflection is a global index permutation
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'p': _P, 'seed': 0} for s in _SIZES],
    representative={'shape': [96, 96, 96], 'p': _P, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
