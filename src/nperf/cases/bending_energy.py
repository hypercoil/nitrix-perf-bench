# -*- coding: utf-8 -*-
"""Registration penalty: ``nitrix.register.bending_energy`` vs numpy / cupy.

The thin-plate (second-order) bending penalty ``mean ‖∇²u‖²`` over a
displacement field ``(*spatial, ndim)`` -- the squared Frobenius norm of the
per-voxel Hessian (a second central diff of each displacement-Jacobian
component), averaged over voxels.  Unlike ``gradient_smoothness`` it leaves a
uniform (affine) flow free, penalising only curvature.  Differentiable.

Warranted comparison: the same central-diff / ``'nearest'`` boundary as
``jacobian_displacement`` applied twice, so the numpy reimplementation (fp64
oracle + CPU floor, verified ~4.1e-7) is the exact-convention target; cupy is
the on-target GPU bar.  Two stencil passes + a reduction (heavier than
``gradient_smoothness``), GPU-pure.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import bending_energy

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_bending_energy,
    displacement_input,
    jacobian_sizes,
    np_bending_energy,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    U = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0),
                           scale=param.get('scale', 0.1))
    jx = jax.block_until_ready(jnp.asarray(U))
    ref = np_bending_energy(U.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(U)
        return (U,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda u: bending_energy(u)),
        'numpy.bending_energy': ('numpy', np_bending_energy),
        'cupy.bending_energy': ('cupy', cupy_bending_energy()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='bending_energy',
    op_qualname='nitrix.register.bending_energy',
    output_independent=False,  # a global mean over the whole field
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'scale': 0.1, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'scale': 0.1, 'seed': 0},
    large_param_points=tuple(
        {'d': d, 'scale': 0.1, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the voxel count N but ~2x gradient_smoothness: the '
        'displacement Jacobian, then a SECOND central-diff of each of its '
        'd*d components (the per-voxel Hessian, ~d*d*d stencil passes) + a '
        'Frobenius reduction. Stencil-heavy, bandwidth-bound, GPU-pure; '
        'HBM ~ N with a larger constant (the (d*d, d) Hessian intermediate '
        'materialises). The brain-scale tier varies the volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
