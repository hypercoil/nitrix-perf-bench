# -*- coding: utf-8 -*-
"""Transform-exp: ``nitrix.geometry.affine_exp`` vs numpy / cupy (batched).

Homogeneous affine transform from its 12 Lie parameters (3-D): the linear block
is ``matrix_exp(A)`` of the row-major gl(3) generator (guaranteeing an
invertible, ``det > 0`` map), translation placed directly.  Batched over ``B``
transforms.  The batched sibling of ``matrix_exp`` (already benched as the
dense expm) wrapped with the translation assembly -- so this case isolates the
batched-wrapper + matrix_exp throughput at the small 4x4 scale.

Warranted comparison: the numpy reimplementation (a bounded-norm Taylor expm of
the same generator + direct translation, verified to fp32 round-off vs nitrix)
is the fp64 oracle + CPU floor; cupy (same body) is the on-target GPU bar.
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import affine_exp

from ._base import BuiltPoint, Case, to_cupy
from ._transforms import affine_params, cupy_affine_exp, np_affine_exp


def _build(param: Dict[str, Any]) -> BuiltPoint:
    b = param['b']
    P = affine_params(b, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(P))
    ref = np_affine_exp(P.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(P)
        return (P,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda p: affine_exp(p, ndim=3)),
        'numpy.affine_exp': ('numpy', np_affine_exp),
        'cupy.affine_exp': ('cupy', cupy_affine_exp()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [1024, 16384, 65536]
_LARGE = [262144, 1048576]

CASE = Case(
    name='affine_exp',
    op_qualname='nitrix.geometry.affine_exp',
    output_independent=True,  # each transform depends only on its own params
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'b': b, 'seed': 0} for b in _SIZES],
    representative={'b': 16384, 'seed': 0},
    large_param_points=tuple({'b': b, 'seed': 0} for b in _LARGE),
    complexity=(
        'O(B) over the batch B, embarrassingly parallel, but heavier per '
        'element than rigid_exp: a matrix_exp (scaling-and-squaring, ~20 3x3 '
        'matmuls) of the gl(3) generator + a direct translation. '
        'Throughput-bound; HBM ~ B. The batch tier varies B. (The 4x4 matrix '
        'is the small-N regime of the matrix_exp case, here batched.)'),
    build=_build,
    rtol=1e-3,
    atol=5e-4,
)
