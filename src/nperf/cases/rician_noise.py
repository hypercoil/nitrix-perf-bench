# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.rician_noise`` vs MONAI / cupy.

Rician (MR magnitude) noise: ``sqrt((x + n_r)**2 + n_i**2)`` with
``n_r, n_i ~ N(0, sigma**2)`` independent -- the magnitude of a complex signal
with independent Gaussian real/imaginary perturbations (reduces to ``|x|`` at
``sigma=0``). Two normal draws + a fused magnitude.

**RNG op -- no cross-framework oracle (`fp64_reference=None`).** As for
`gaussian_noise`: nitrix draws from the jax PRNG (fixed key -> deterministic +
drift-stable), the refs from their own RNG; the ratio is a task-level
wall-clock comparison and the distribution is validated in tests. cupy is the
GPU headline ref; MONAI `RandRicianNoise` (fixed std, absolute) is the CPU
community baseline -- it implements the same `sqrt((x+n1)^2 + n2^2)` model.
Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import rician_noise

from ._augment import (
    augment_input,
    cupy_rician_noise,
    monai_rician_noise,
    np_rician_noise,
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
        'nitrix-jax': ('jax', lambda x, k: rician_noise(x, k, sigma=sigma)),
        'numpy.rician_noise': ('numpy', np_rician_noise(sigma, seed)),
        'cupy.rician_noise': ('cupy', cupy_rician_noise(sigma, seed)),
        'monai.RandRicianNoise': ('monai', monai_rician_noise(sigma)),
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


# (cube side): two n^3 normal draws + a fused magnitude; memory-bound.
_SIZES = [64, 96, 128]

CASE = Case(
    name='rician_noise',
    op_qualname='nitrix.augment.rician_noise',
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
