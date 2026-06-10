# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.partialcorr`` vs numpy / cupy.

Partial correlation: partial covariance normalised by the geometric mean of
its absolute diagonal, i.e. ``-P_ij / sqrt(P_ii · P_jj)`` with unit diagonal.
nitrix (jax) vs a numpy construction (CPU floor) + a CuPy construction (GPU
ref), scored against an fp64 oracle.  See ``cases/_precision.py`` for the exact
construction (the sign flip + ``|diag|`` normalisation are the subtleties) and
the GPU / cuSolver story.  Ratio vs ``numpy.partialcorr``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import partialcorr

from ._base import BuiltPoint, Case, to_cupy
from ._precision import (
    cupy_inv_family,
    inv_family,
    nilearn_conn,
    precision_input,
)

_KIND = 'partialcorr'


def _build(param: Dict[str, Any]) -> BuiltPoint:
    c, obs = param['c'], param['obs']
    X = precision_input(c, obs, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = inv_family(X.astype(np.float64), _KIND, np)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: partialcorr(x)),
        'numpy.partialcorr': ('numpy', lambda x: inv_family(x, _KIND, np)),
        'nilearn.partial_correlation': (  # community-standard floor (exact)
            'nilearn', nilearn_conn(_KIND)),
        'cupy.partialcorr': ('cupy', cupy_inv_family(_KIND)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (variables, observations): obs > c keeps the covariance non-singular; cost is
# ~ c·obs (cov) + c^3 (inverse).
_SHAPES = [(128, 1024), (256, 2048), (512, 4096)]
_LARGE = [(1024, 4096), (2048, 8192)]  # brain-parcel scale

CASE = Case(
    name='partialcorr',
    op_qualname='nitrix.stats.partialcorr',
    output_independent=False,  # inverse couples every entry
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'obs': o, 'seed': 0} for (c, o) in _SHAPES],
    representative={'c': 256, 'obs': 2048, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'obs': o, 'seed': 0} for (c, o) in _LARGE),
    complexity=(
        'precision (cov O(c^2*obs) + inverse O(c^3)) then normalising by the '
        'geometric mean of the diagonal -- the inverse dominates at '
        'brain-parcel c; HBM ~ c^2. Same GPU inverse as precision (a measured '
        'scale-WIN: nitrix consumed-inv beats the cupy GPU inv increasingly '
        'with c). The size tier varies c to parcel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
