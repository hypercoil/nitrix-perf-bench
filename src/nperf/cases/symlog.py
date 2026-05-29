# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.symlog`` vs scipy / cupy.

SPD matrix logarithm.  nitrix (jax, eigh-based) vs the textbook
``scipy.linalg.logm`` (CPU floor) and a CuPy eigh-based GPU reference, all
scored against an fp64 eigh-based oracle on a well-conditioned SPD input.

**nitrix runs this on the GPU** (jitted; the matrix function *consumes* the
eigh, which XLA lowers off the cuSOLVER path that breaks cupy / bare eigh at
d≥256 -- see ``cases/_spd.py``).  The **cupy GPU ref fails at d≥256** (recorded
``gpu_solver_unavailable``), so the GPU-vs-GPU bar holds only at d=64;
scipy.linalg.logm is the CPU floor.  Ratio vs ``scipy.linalg.logm``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from nitrix.linalg import symlog

from ._base import BuiltPoint, Case, to_cupy
from ._spd import cupy_matrix_fn, eig_matrix_fn, spd_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    A = spd_input(d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = eig_matrix_fn(A.astype(np.float64), np.log)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: symlog(a)),
        'scipy.linalg.logm': ('scipy', lambda a: np.real(sla.logm(a))),
        'cupy.eigh_logm': ('cupy', cupy_matrix_fn('log')),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [64, 256, 512]

CASE = Case(
    name='symlog',
    op_qualname='nitrix.linalg.symlog',
    output_independent=False,  # every output entry couples all eigenpairs
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
