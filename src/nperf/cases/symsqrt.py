# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.symsqrt`` vs scipy / cupy.

SPD matrix square root, sibling of ``symlog`` (see it + ``cases/_spd.py`` for
the eigh-family / GPU-blocker story).  nitrix (jax, eigh-based) vs
``scipy.linalg.sqrtm`` (CPU floor) + a CuPy eigh-based GPU reference, scored
against an fp64 eigh-based oracle.  nitrix runs this on the GPU (jitted; the
matrix function consumes the eigh, dodging the cuSOLVER path -- see
``cases/_spd.py``); the cupy GPU ref fails at d≥256.  Ratio via
``--reference scipy.linalg.sqrtm``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from nitrix.linalg import symsqrt

from ._base import BuiltPoint, Case, to_cupy
from ._spd import cupy_matrix_fn, eig_matrix_fn, spd_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    A = spd_input(d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = eig_matrix_fn(A.astype(np.float64), np.sqrt)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: symsqrt(a)),
        'scipy.linalg.sqrtm': ('scipy', lambda a: np.real(sla.sqrtm(a))),
        'cupy.eigh_sqrtm': ('cupy', cupy_matrix_fn('sqrt')),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [64, 256, 512]

CASE = Case(
    name='symsqrt',
    op_qualname='nitrix.linalg.symsqrt',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 256, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
