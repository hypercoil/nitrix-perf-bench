# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.symexp`` vs scipy / cupy.

Symmetric matrix exponential ``V·diag(exp(λ))·Vᵀ`` -- the inverse of ``symlog``
(maps the tangent space back into the SPD cone). nitrix (jax, eigh-based) vs
the textbook ``scipy.linalg.expm`` (CPU floor; for a symmetric matrix it equals
the eigh-based exp) + a CuPy eigh-based GPU reference, scored against an fp64
eigh-based oracle on a well-conditioned SPD input.

Like symlog/symsqrt/sympower, the matrix function *consumes* the eigh, which
XLA lowers to a path that runs on the GPU on this box, where bare / cupy eigh
does not at d≥256 (a cuSOLVER-class issue observed here -- cause/scope
uncharacterised, not a portable claim; see ``cases/_spd.py``). Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from nitrix.linalg import symexp

from ._base import BuiltPoint, Case, to_cupy
from ._spd import cupy_matrix_fn, eig_matrix_fn, spd_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    A = spd_input(d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = eig_matrix_fn(A.astype(np.float64), np.exp)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: symexp(a)),
        'scipy.linalg.expm': ('scipy', lambda a: np.real(sla.expm(a))),
        'cupy.eigh_expm': ('cupy', cupy_matrix_fn('exp')),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [64, 256, 512]

CASE = Case(
    name='symexp',
    op_qualname='nitrix.linalg.symexp',
    output_independent=False,  # every output entry couples all eigenpairs
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
