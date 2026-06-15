# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.gamma_contrast`` vs MONAI / cupy.

A gamma tone curve within the per-tensor min/max bracket (the FM-pretraining
intensity augmentation): ``normed = clip((x-lo)/span, 0, 1)`` then
``out = normed**gamma * span + lo``. Memory-bound elementwise, but the min/max
bracket is a global reduction (so `output_independent=False`).

References: nitrix (jax) vs the numpy reimplementation (fp64 oracle, nitrix's
`eps=1e-8` matched), a **cupy GPU reference** (the on-target headline bar), and
**MONAI `AdjustContrast`** -- the de-facto community augmentation toolkit, the
community baseline. nitrix matches MONAI to ~7e-8 in fp32, so it is a clean
apples-to-apples reference (unlike e.g. MONAI `GibbsNoise`, which models a
different artifact). MONAI runs CPU-only for now (the `monai` provider; a GPU
MONAI env is a tracked follow-up); the GPU headline is read from nitrix-jax vs
cupy. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import gamma_contrast

from ._augment import (
    augment_input,
    cupy_gamma,
    monai_adjust_contrast,
    np_gamma,
)
from ._base import BuiltPoint, Case, to_cupy

_GAMMA = 0.7  # < 1: expand dark-value dynamic range (a contrast boost)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    gamma = float(param.get('gamma', _GAMMA))
    X = augment_input(param['shape'], param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_gamma(gamma)(X.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        # numpy oracle + MONAI wrapper both take the host array.
        return (X,) if framework in ('numpy', 'monai') else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: gamma_contrast(x, gamma)),
        'numpy.gamma_contrast': ('numpy', np_gamma(gamma)),  # CPU floor/oracle
        'cupy.gamma_contrast': ('cupy', cupy_gamma(gamma)),  # GPU headline ref
        'monai.AdjustContrast': ('monai', monai_adjust_contrast(gamma)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube side): elementwise + a global min/max reduce; cost ~ n^3.
_SIZES = [64, 96, 128]

CASE = Case(
    name='gamma_contrast',
    op_qualname='nitrix.augment.gamma_contrast',
    output_independent=False,  # the min/max bracket couples the whole tensor
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'gamma': _GAMMA, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'gamma': _GAMMA, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
