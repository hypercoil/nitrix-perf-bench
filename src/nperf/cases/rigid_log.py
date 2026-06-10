# -*- coding: utf-8 -*-
"""Transform-exp: ``nitrix.geometry.rigid_log`` vs numpy / cupy (batched).

The inverse of ``rigid_exp``: recover the 6 Lie parameters of a rigid
homogeneous matrix -- principal SO(3) log of the rotation block (axis-angle) +
the translation column.  Batched over ``B`` transforms.

Warranted comparison: the numpy reimplementation (principal axis-angle log +
translation, verified ~1e-7 vs nitrix and a clean exp/log round-trip) is the
fp64 oracle + CPU floor; cupy (same body) is the on-target GPU bar.  The inputs
are valid rigid matrices from ``rigid_exp`` at bounded ``ω`` (|ω| < ~1.5 rad),
staying away from the ``θ = π`` log singularity (axis ill-defined there) -- the
well-posed interior, the regime users hit.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import rigid_log

from ._base import BuiltPoint, Case, to_cupy
from ._transforms import cupy_rigid_log, np_rigid_log, rigid_matrices


def _build(param: Dict[str, Any]) -> BuiltPoint:
    b = param['b']
    M = rigid_matrices(b, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(M))
    ref = np_rigid_log(M.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(M)
        return (M,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: rigid_log(m, ndim=3)),
        'numpy.rigid_log': ('numpy', np_rigid_log),
        'cupy.rigid_log': ('cupy', cupy_rigid_log()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [1024, 16384, 65536]
_LARGE = [262144, 1048576]

CASE = Case(
    name='rigid_log',
    op_qualname='nitrix.geometry.rigid_log',
    output_independent=True,  # each transform depends only on its own matrix
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'b': b, 'seed': 0} for b in _SIZES],
    representative={'b': 16384, 'seed': 0},
    large_param_points=tuple({'b': b, 'seed': 0} for b in _LARGE),
    complexity=(
        'O(B) over the batch B, embarrassingly parallel: principal SO(3) log '
        '(trace -> angle, off-diagonals -> axis) + the translation column per '
        'transform, tiny 4x4 matrices. Throughput-bound; HBM ~ B. The batch '
        'tier varies B.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
