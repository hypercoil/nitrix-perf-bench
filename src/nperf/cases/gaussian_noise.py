# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.gaussian_noise`` vs MONAI / cupy.

Additive i.i.d. ``N(0, sigma**2)`` per element (`x + sigma * normal(key)`) --
the simplest intensity augmentation. Memory-bound: draw N normals + a fused
multiply-add.

**RNG op -- no cross-framework oracle (`fp64_reference=None`).** nitrix draws
from the jax PRNG (a fixed key here, so it is deterministic + drift-stable),
the references from their own RNG, so the *samples* differ -- there is no
elementwise oracle. The perf ratio is a fair **task-level wall-clock**
comparison (generate N draws + multiply-add, same sigma); the distribution is
validated in `tests/test_augment_cases.py`. cupy is the on-target GPU headline
ref (jax vs cupy RNG+add on GPU); MONAI `RandGaussianNoise` (fixed std) is the
CPU community baseline. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import gaussian_noise

from ._augment import (
    augment_input,
    cupy_gaussian_noise,
    monai_gaussian_noise,
    np_gaussian_noise,
)
from ._base import BuiltPoint, Case, to_cupy

_SIGMA = 0.1


def _build(param: Dict[str, Any]) -> BuiltPoint:
    sigma = float(param.get('sigma', _SIGMA))
    seed = int(param.get('seed', 0))
    X = augment_input(param['shape'], seed)
    jx = jax.block_until_ready(jnp.asarray(X))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        if framework == 'jax':
            return (jx, key)
        return (X,)  # numpy + monai refs draw their own noise

    baselines = {
        'nitrix-jax': ('jax', lambda x, k: gaussian_noise(x, k, sigma=sigma)),
        'numpy.gaussian_noise': ('numpy', np_gaussian_noise(sigma, seed)),
        'cupy.gaussian_noise': ('cupy', cupy_gaussian_noise(sigma, seed)),
        'monai.RandGaussianNoise': ('monai', monai_gaussian_noise(sigma)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op: nitrix draws from jax PRNG, the refs from '
                      'their own RNG -- no elementwise oracle. The ratio is a '
                      'task-level wall-clock comparison; distribution checked '
                      'in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): draw n^3 normals + a multiply-add; memory-bound.
_SIZES = [64, 96, 128]

CASE = Case(
    name='gaussian_noise',
    op_qualname='nitrix.augment.gaussian_noise',
    output_independent=True,  # each element is perturbed independently
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'sigma': _SIGMA, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'sigma': _SIGMA, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
