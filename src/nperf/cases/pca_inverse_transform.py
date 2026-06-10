# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.pca_inverse_transform`` (numpy/cupy).

Reconstruction from PCA coordinates: ``X_hat = Z @ components + mean`` -- the
inverse twin of ``pca_transform`` and, like it, a pure BLAS matmul (no eigh, no
cuSOLVER fallback).  System-under-test is nitrix on jax; the CPU floor is
the natural ``numpy`` matmul and the GPU twin is the same matmul on ``cupy``.
The coordinates ``Z`` and the basis (``components``, ``mean``) are computed
**once** (``_pca.np_basis`` + project) and shared verbatim by every framework,
so the output is unique and the fp64 oracle is the same reconstruction in
double.  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import pca_inverse_transform

from ._base import BuiltPoint, Case, to_cupy
from ._pca import cupy_inverse, np_basis, np_inverse, np_transform, pca_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d, k = param['n'], param['d'], param['k']
    X = pca_input(n, d, k, param.get('seed', 0))
    components, mean = np_basis(X, k)  # fixed fp32 basis, shared by all
    Z = np_transform(X, components, mean).astype(np.float32)  # (n, k) coords
    jz = jax.block_until_ready(jnp.asarray(Z))
    jc = jax.block_until_ready(jnp.asarray(components))
    jm = jax.block_until_ready(jnp.asarray(mean))

    # fp64 oracle: the same reconstruction in double on the same values.
    ref = np_inverse(Z.astype(np.float64), components.astype(np.float64),
                     mean.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(Z, components, mean)
        if framework == 'numpy':
            return (Z, components, mean)
        return (jz, jc, jm)

    baselines = {
        'nitrix-jax': ('jax', lambda z, c, m: pca_inverse_transform(z, c, m)),
        'numpy.matmul': ('numpy', np_inverse),  # CPU floor
        'cupy.matmul': ('cupy', cupy_inverse),  # GPU twin
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features, components): n is the scale axis; the (n, k) @ (k, d)
# matmul is BLAS-bound and GPU-friendly (the inverse twin of pca_transform).
_SHAPES = [(8192, 256, 32), (16384, 512, 64), (8192, 1024, 64)]
# Brain-voxel scale: n up to ~131072 samples (a whole-brain reconstruction).
_LARGE = [(65536, 1024, 64), (131072, 512, 64)]

CASE = Case(
    name='pca_inverse_transform',
    op_qualname='nitrix.stats.pca_inverse_transform',
    output_independent=True,  # row i of X_hat depends only on row i of Z
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'k': k, 'seed': 0}
                  for (n, d, k) in _SHAPES],
    representative={'n': 16384, 'd': 512, 'k': 64, 'seed': 0},
    large_param_points=tuple(
        {'n': n, 'd': d, 'k': k, 'seed': 0} for (n, d, k) in _LARGE),
    complexity=(
        'Z @ components + mean: O(n * d * k) -- one BLAS-class matmul, the '
        'GPU-friendly regime (no eigh, so no cuSOLVER fallback; contrast '
        'pca_fit). HBM ~ n*d (the output dominates) + n*k (input). The size '
        'tier varies n to whole-brain voxel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
