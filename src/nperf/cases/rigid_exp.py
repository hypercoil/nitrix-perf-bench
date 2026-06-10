# -*- coding: utf-8 -*-
"""Transform-exp: ``nitrix.geometry.rigid_exp`` vs numpy / cupy (batched).

Homogeneous rigid transform from its 6 Lie parameters (3-D): axis-angle ``ω``
-> Rodrigues SO(3) rotation, translation placed directly.  Batched over a
cohort of ``B`` transforms (the bench axis) -- tiny 4x4 matrices, so this is an
embarrassingly-parallel throughput op.

Warranted comparison: the numpy reimplementation (Rodrigues + direct
translation, verified ~5e-7) is the fp64 oracle + CPU floor; cupy (the
same body) is the on-target GPU bar.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import rigid_exp

from ._base import BuiltPoint, Case, to_cupy
from ._transforms import cupy_rigid_exp, np_rigid_exp, rigid_params


def _build(param: Dict[str, Any]) -> BuiltPoint:
    b = param['b']
    P = rigid_params(b, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(P))
    ref = np_rigid_exp(P.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(P)
        return (P,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda p: rigid_exp(p, ndim=3)),
        'numpy.rigid_exp': ('numpy', np_rigid_exp),
        'cupy.rigid_exp': ('cupy', cupy_rigid_exp()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [1024, 16384, 65536]
_LARGE = [262144, 1048576]

CASE = Case(
    name='rigid_exp',
    op_qualname='nitrix.geometry.rigid_exp',
    output_independent=True,  # each transform depends only on its own params
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'b': b, 'seed': 0} for b in _SIZES],
    representative={'b': 16384, 'seed': 0},
    large_param_points=tuple({'b': b, 'seed': 0} for b in _LARGE),
    complexity=(
        'O(B) over the batch B, embarrassingly parallel: Rodrigues SO(3) exp '
        '(a few 3x3 matmuls) + a direct translation per transform, tiny 4x4 '
        'matrices. Throughput-bound; launch-bound at small B (GPU favoured as '
        'B grows). HBM ~ B. The batch tier varies B (cohort / per-voxel '
        'local-affine field scale).'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
