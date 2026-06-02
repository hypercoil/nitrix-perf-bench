# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.rbf_kernel`` vs sklearn / cupy.

The RBF (Gaussian) kernel ``exp(-gamma · ‖x - y‖²)`` of a feature matrix:
nitrix (jax) vs ``sklearn.metrics.pairwise.rbf_kernel`` (the canonical CPU
floor) + a CuPy GPU reference (Gram-matrix derived; see ``cases/_kernels.py``),
scored against an fp64 oracle (scipy ``cdist`` -- direct squared distances, no
cancellation -- then ``exp``).  Matmul/broadcast based, so GPU-pure (no
cuSolver): a clean apples-to-apples GPU bar.  ``gamma = 1/d`` (sklearn's
default, pinned explicitly so all impls agree).  Ratio vs the sklearn kernel.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.linalg import rbf_kernel
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf_kernel

from ._base import BuiltPoint, Case, to_cupy
from ._kernels import cupy_kernel, kernel_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d = param['n'], param['d']
    X = kernel_input(n, d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    gamma = 1.0 / d

    Xd = X.astype(np.float64)
    d2 = cdist(Xd, Xd, 'sqeuclidean')  # direct (no cancellation)
    ref = np.exp(-gamma * d2)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: rbf_kernel(x, gamma=gamma)),
        'sklearn.rbf_kernel': (
            'sklearn', lambda x: sk_rbf_kernel(x, gamma=gamma)),
        'cupy.rbf_kernel': ('cupy', cupy_kernel('rbf', gamma=gamma)),  # GPU
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features): cost ~ n² · d (matmul-bound); output is n x n.
_SHAPES = [(512, 64), (2048, 64), (4096, 64)]

CASE = Case(
    name='rbf_kernel',
    op_qualname='nitrix.linalg.rbf_kernel',
    # K[i, j] couples rows i and j; not element-wise independent, but the fp64
    # oracle is computed in full (cheap), so this is documentary (annex §C).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'seed': 0} for (n, d) in _SHAPES],
    representative={'n': 2048, 'd': 64, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
