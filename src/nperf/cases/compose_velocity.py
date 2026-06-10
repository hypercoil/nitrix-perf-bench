# -*- coding: utf-8 -*-
"""Geometry SVF algebra: ``nitrix.geometry.compose_velocity`` vs numpy / cupy.

The BCH composition of two stationary velocity fields (the log-domain update
whose exponential is ``exp(v) ∘ exp(u)``).  Benched at **``order=2``** -- the
first Baker-Campbell-Hausdorff correction ``z = v + u + ½ [v, u]`` with the Lie
bracket ``[v, u] = (v·∇)u - (u·∇)v`` -- the branch with real compute (two
displacement Jacobians + two contractions).  ``order=1`` is a trivial
elementwise ``v + u`` (the additive update most demons implementations default
to) -- a memory-bound add with no kernel story, so the order-2 bracket is the
informative branch.

Warranted comparison: the numpy reimplementation uses the same central-diff
Jacobian convention (via ``_jacobian``) for the bracket -- verified ~4.7e-8 vs
nitrix -- the fp64 oracle + CPU floor; cupy is the GPU bar.  Stencil +
contraction, GPU-pure.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import compose_velocity

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_compose_velocity,
    displacement_input,
    jacobian_sizes,
    np_compose_velocity,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    shape = jacobian_sizes(d)
    v = displacement_input(shape, _NDIM, param.get('seed', 0), scale=0.1)
    u = displacement_input(shape, _NDIM, param.get('seed', 0) + 1, scale=0.1)
    vj = jax.block_until_ready(jnp.asarray(v))
    uj = jax.block_until_ready(jnp.asarray(u))
    ref = np_compose_velocity(v.astype('float64'), u.astype('float64'))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(v, u)
        return (v, u) if framework == 'numpy' else (vj, uj)

    baselines = {
        'nitrix-jax': ('jax', lambda a, b: compose_velocity(a, b, order=2)),
        'numpy.compose_velocity': ('numpy', np_compose_velocity),
        'cupy.compose_velocity': ('cupy', cupy_compose_velocity()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='compose_velocity',
    op_qualname='nitrix.geometry.compose_velocity',
    output_independent=False,  # the bracket's Jacobians couple neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the voxel count N (order=2): two displacement Jacobians '
        '(central-diff stencils) + two ...ij,...j contractions + the add. '
        'Stencil + contraction, memory-bandwidth-bound, GPU-pure; HBM ~ N '
        '(the Jacobian intermediates). order=1 is a trivial elementwise add. '
        'The brain-scale tier varies the volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
