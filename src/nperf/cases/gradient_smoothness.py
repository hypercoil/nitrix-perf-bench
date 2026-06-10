# -*- coding: utf-8 -*-
"""Registration penalty: ``nitrix.register.gradient_smoothness`` vs numpy/cupy.

The diffusion (first-order) smoothness penalty ``mean ‖∇u‖²`` over a
displacement field ``(*spatial, ndim)`` -- the squared Frobenius norm of the
displacement Jacobian ``∇u = J - I``, averaged over voxels.  A differentiable
training-loss regulariser.

Warranted comparison: nitrix builds on the same central-diff / ``'nearest'``
boundary as ``jacobian_displacement`` (denominator ``2·spacing`` even at the
edge cell -- the voxelmorph convention), so the numpy reimpl (the fp64
oracle + CPU floor, verified ~1.6e-7) is the exact-convention target, **not**
``numpy.gradient``; cupy is the on-target GPU bar.  Stencil + reduction,
GPU-pure (no solver).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import gradient_smoothness

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_gradient_smoothness,
    displacement_input,
    jacobian_sizes,
    np_gradient_smoothness,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    U = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0),
                           scale=param.get('scale', 0.1))
    jx = jax.block_until_ready(jnp.asarray(U))
    ref = np_gradient_smoothness(U.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(U)
        return (U,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda u: gradient_smoothness(u)),
        'numpy.gradient_smoothness': ('numpy', np_gradient_smoothness),
        'cupy.gradient_smoothness': ('cupy', cupy_gradient_smoothness()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='gradient_smoothness',
    op_qualname='nitrix.register.gradient_smoothness',
    output_independent=False,  # a global mean over the whole field
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'scale': 0.1, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'scale': 0.1, 'seed': 0},
    large_param_points=tuple(
        {'d': d, 'scale': 0.1, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the voxel count N: one roll-based central-diff pass (the '
        'displacement Jacobian) + a Frobenius reduction. Stencil + reduction, '
        'memory-bandwidth-bound and GPU-pure (no solver); HBM ~ N (a few '
        'd-component field copies). The brain-scale tier varies the volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
