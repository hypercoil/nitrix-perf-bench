# -*- coding: utf-8 -*-
"""Registration penalty: ``nitrix.register.jacobian_folding_penalty``.

The folding penalty ``mean relu(-det J)`` of the deformation Jacobian
``J = I + ∇u`` over a displacement field ``(*spatial, ndim)`` -- zero where the
map is locally orientation-preserving (``det J > 0``), growing with the
magnitude of any fold (``det J <= 0``).  The invertibility QA term a warp adds
to its loss.  Differentiable (sub-gradient at the relu kink).

Warranted comparison: the same central-diff / ``'nearest'`` boundary (closed-
form Sarrus determinant for d=3), so the numpy reimplementation is the fp64
oracle + CPU floor; cupy is the on-target GPU bar.  The input is scaled up
(``scale=1.0``, a deliberately folding warp) so the ``relu(-det)`` branch is
actually exercised -- a realistic small warp folds nowhere and the penalty is a
degenerate 0 (the *compute* is branch-independent: det + relu over every voxel
regardless).  Verified ~6e-9 at this scale.  Stencil + reduction, GPU-pure.
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import jacobian_folding_penalty

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_folding,
    displacement_input,
    jacobian_sizes,
    np_folding,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    U = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0),
                           scale=param.get('scale', 1.0))
    jx = jax.block_until_ready(jnp.asarray(U))
    ref = np_folding(U.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(U)
        return (U,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda u: jacobian_folding_penalty(u)),
        'numpy.jacobian_folding_penalty': ('numpy', np_folding),
        'cupy.jacobian_folding_penalty': ('cupy', cupy_folding()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='jacobian_folding_penalty',
    op_qualname='nitrix.register.jacobian_folding_penalty',
    output_independent=False,  # a global mean over the whole field
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'scale': 1.0, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'scale': 1.0, 'seed': 0},
    large_param_points=tuple(
        {'d': d, 'scale': 1.0, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(N) over the voxel count N: one roll-based central-diff pass (the '
        'Jacobian) + a closed-form Sarrus determinant + relu + a mean. '
        'Stencil + reduction, memory-bandwidth-bound, GPU-pure; HBM ~ N. The '
        'relu makes the *value* data-dependent but the *cost* is not (det + '
        'relu run on every voxel). The brain-scale tier varies the volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-5,
)
