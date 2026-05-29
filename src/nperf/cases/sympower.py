# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.sympower`` vs scipy / cupy.

SPD matrix power ``A**p`` (here ``p = 0.75``), sibling of the ``symlog`` /
``symsqrt`` cases (see ``cases/_spd.py`` for the eigh-family / GPU-blocker
story).  nitrix (jax, eigh-based) vs ``scipy.linalg.fractional_matrix_power``
(CPU floor) + a CuPy eigh-based GPU reference, scored against an fp64
eigh-based oracle.  nitrix runs this on the GPU (jitted; the matrix function
consumes the eigh, dodging the cuSOLVER path -- see ``cases/_spd.py``); the
cupy GPU ref fails at d≥256.  Ratio via
``--reference scipy.linalg.fractional_matrix_power``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from nitrix.linalg import sympower

from ._base import BuiltPoint, Case, to_cupy
from ._spd import cupy_matrix_fn, eig_matrix_fn, spd_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d, p = param['d'], param['power']
    A = spd_input(d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(A))
    ref = eig_matrix_fn(A.astype(np.float64), lambda w: w ** p)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(A)
        return (A,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: sympower(a, power=p)),
        'scipy.linalg.fractional_matrix_power': (
            'scipy', lambda a: np.real(sla.fractional_matrix_power(a, p))),
        'cupy.eigh_matpow': ('cupy', cupy_matrix_fn('power', p)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [64, 256, 512]

CASE = Case(
    name='sympower',
    op_qualname='nitrix.linalg.sympower',
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'power': 0.75, 'seed': 0} for d in _SIZES],
    representative={'d': 256, 'power': 0.75, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
