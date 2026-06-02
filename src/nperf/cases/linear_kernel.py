# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.linear_kernel`` vs sklearn / cupy.

The linear kernel ``X @ Xᵀ`` (Gram matrix) of a feature matrix: nitrix (jax) vs
``sklearn.metrics.pairwise.linear_kernel`` (the canonical CPU floor) + a CuPy
GPU reference (``X @ Xᵀ``; see ``cases/_kernels.py``), scored against an fp64
oracle (``X @ Xᵀ`` in double).  Pure matmul, so GPU-pure (no cuSolver): a clean
apples-to-apples GPU bar -- a BLAS GEMM at heart, the friendliest case for the
GPU.  Ratio vs the sklearn kernel.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.linalg import linear_kernel
from sklearn.metrics.pairwise import linear_kernel as sk_linear_kernel

from ._base import BuiltPoint, Case, to_cupy
from ._kernels import cupy_kernel, kernel_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d = param['n'], param['d']
    X = kernel_input(n, d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))

    Xd = X.astype(np.float64)
    ref = Xd @ Xd.T  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: linear_kernel(x)),
        'sklearn.linear_kernel': ('sklearn', lambda x: sk_linear_kernel(x)),
        'cupy.linear_kernel': ('cupy', cupy_kernel('linear')),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features): cost ~ n² · d (a single GEMM); output is n x n.
_SHAPES = [(512, 64), (2048, 64), (4096, 64)]

CASE = Case(
    name='linear_kernel',
    op_qualname='nitrix.linalg.linear_kernel',
    # K[i, j] = ⟨xᵢ, xⱼ⟩ couples rows i and j; not element-wise independent,
    # but the fp64 oracle is computed in full (cheap) -- documentary (§C).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'seed': 0} for (n, d) in _SHAPES],
    representative={'n': 2048, 'd': 64, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
