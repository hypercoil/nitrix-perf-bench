# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.precision`` vs numpy / cupy.

Inverse covariance ``inv(cov(X))``.  nitrix (jax) vs a numpy inverse-covariance
(CPU floor) + a CuPy inverse-covariance (GPU ref), scored against an fp64
oracle.  See ``cases/_precision.py`` for the construction and the GPU /
cuSolver story (on this box nitrix's jitted consumed-inv runs on the GPU where
the cupy ref's raw ``cupy.linalg.inv`` fails at large ``c`` -- a cuSOLVER-class
issue observed here, cause/scope uncharacterised).  Ratio vs
``numpy.inv_cov``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import precision

from ._base import BuiltPoint, Case, to_cupy
from ._precision import (
    cupy_inv_family,
    inv_family,
    nilearn_conn,
    precision_input,
)

_KIND = 'precision'


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
        'nitrix-jax': ('jax', lambda x: precision(x)),
        'numpy.inv_cov': ('numpy', lambda x: inv_family(x, _KIND, np)),
        'nilearn.precision': (  # community-standard floor (EmpiricalCov)
            'nilearn', nilearn_conn(_KIND)),
        'cupy.inv_cov': ('cupy', cupy_inv_family(_KIND)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (variables, observations): obs > c keeps the covariance non-singular; cost is
# ~ c·obs (cov) + c^3 (inverse).
_SHAPES = [(128, 1024), (256, 2048), (512, 4096)]
# Brain-parcel scale: c up to ~2048 parcels (dense voxel connectivity's C x C
# is infeasible; parcellations top out here). obs > c keeps cov non-singular.
_LARGE = [(1024, 4096), (2048, 8192)]

CASE = Case(
    name='precision',
    op_qualname='nitrix.stats.precision',
    output_independent=False,  # inverse couples every entry
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'obs': o, 'seed': 0} for (c, o) in _SHAPES],
    representative={'c': 256, 'obs': 2048, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'obs': o, 'seed': 0} for (c, o) in _LARGE),
    complexity=(
        'cov is O(c^2 * obs); the INVERSE is O(c^3) and dominates at '
        'brain-parcel c. HBM ~ c^2. MEASURED (L4): nitrix jits a consumed-inv '
        'that scales WELL on the GPU -- it beats the cupy GPU '
        'inverse-covariance by a GROWING margin (2.35x at c=256 -> 11.5x at '
        'c=2048), and cupy ran across the range (no cuSOLVER failure observed '
        'up to c=2048). numpy/nilearn are the CPU floor (slow at c>=1024). A '
        'scale-WIN; the size tier varies c to parcel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
