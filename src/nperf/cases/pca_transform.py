# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.pca_transform`` vs numpy / cupy.

Projection onto a **pre-fitted** PCA basis: ``Z = (X - mean) @ components^T``,
a pure BLAS matmul, the GPU-friendly regime (no eigh, so none of the
``pca_fit`` cuSOLVER fallback).  System-under-test is nitrix on jax; the
CPU floor is the natural ``numpy`` matmul and the GPU twin is the same matmul
on ``cupy`` (apples-to-apples on device).  The basis (``components``, ``mean``)
is computed **once** (``_pca.np_basis``) and shared verbatim by all frameworks,
so there is no sign ambiguity -- the output is unique and the fp64 oracle the
same projection in double.  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import pca_transform

from ._base import BuiltPoint, Case, to_cupy
from ._pca import cupy_transform, np_basis, np_transform, pca_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d, k = param['n'], param['d'], param['k']
    X = pca_input(n, d, k, param.get('seed', 0))
    components, mean = np_basis(X, k)  # fixed fp32 basis, shared by all
    jx = jax.block_until_ready(jnp.asarray(X))
    jc = jax.block_until_ready(jnp.asarray(components))
    jm = jax.block_until_ready(jnp.asarray(mean))

    # fp64 oracle: the same projection in double on the same values.
    ref = np_transform(X.astype(np.float64), components.astype(np.float64),
                       mean.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, components, mean)
        if framework == 'numpy':
            return (X, components, mean)
        return (jx, jc, jm)

    baselines = {
        'nitrix-jax': ('jax', lambda x, c, m: pca_transform(x, c, m)),
        'numpy.matmul': ('numpy', np_transform),  # CPU floor
        'cupy.matmul': ('cupy', cupy_transform),  # GPU twin
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features, components): n is the scale axis (whole-brain voxels as
# samples); the (n, d) @ (d, k) matmul is BLAS-bound and GPU-friendly.
_SHAPES = [(8192, 256, 32), (16384, 512, 64), (8192, 1024, 64)]
# Brain-voxel scale: n up to ~131072 samples (a whole-brain voxel projection).
_LARGE = [(65536, 1024, 64), (131072, 512, 64)]

CASE = Case(
    name='pca_transform',
    op_qualname='nitrix.stats.pca_transform',
    output_independent=True,  # row i of Z depends only on row i of X
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'k': k, 'seed': 0}
                  for (n, d, k) in _SHAPES],
    representative={'n': 16384, 'd': 512, 'k': 64, 'seed': 0},
    large_param_points=tuple(
        {'n': n, 'd': d, 'k': k, 'seed': 0} for (n, d, k) in _LARGE),
    complexity=(
        '(X - mean) @ components^T: O(n * d * k) -- one BLAS-class matmul, '
        'the GPU-friendly regime (no eigh, so no cuSOLVER fallback; contrast '
        'pca_fit). HBM ~ n*d (the input dominates) + n*k (output). The size '
        'tier varies n to whole-brain voxel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
