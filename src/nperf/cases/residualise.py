# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.linalg.residualise`` vs ``numpy`` lstsq.

The fMRI confound-regression workload: residualise ``Y`` (``V`` voxels ×
``N`` TRs) against a design ``X`` (``K`` confounds × ``N``).  nitrix uses the
Cholesky normal-equations path; the reference is the textbook ``numpy.linalg.
lstsq`` solve-then-project (all voxels in one call).  The least-squares
residual is unique, so both agree with the fp64 oracle regardless of method.
Ratio via ``--reference numpy.linalg.lstsq``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.linalg import residualise

from ._base import BuiltPoint, Case, to_cupy
from ._clean import nilearn_clean


def _np_residualise(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    '''lstsq solve-then-project: β = argmin ||Xᵀβ − Yᵀ||; residual Y − proj.'''
    x_t = X.T  # (N, K)
    betas, _, _, _ = np.linalg.lstsq(x_t, Y.T, rcond=None)
    return Y - (x_t @ betas).T


def _cupy_residualise(Y: Any, X: Any) -> Any:
    '''The same lstsq solve-then-project on the GPU (cuSOLVER); cupy lazy.'''
    import cupy as cp

    x_t = X.T
    betas = cp.linalg.lstsq(x_t, Y.T, rcond=None)[0]
    return Y - (x_t @ betas).T


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, n, k = param['V'], param['N'], param['K']
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal((k, n)).astype(np.float32)
    Y = rng.standard_normal((v, n)).astype(np.float32)
    jx_x = jax.block_until_ready(jnp.asarray(X))
    jx_y = jax.block_until_ready(jnp.asarray(Y))

    # fp64 oracle: the same solve-then-project in double.
    ref = _np_residualise(Y.astype(np.float64), X.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        # residualise(Y, X): the (Y, X) order both run_fns take.
        if framework == 'cupy':
            return to_cupy(Y, X)
        return (Y, X) if framework == 'numpy' else (jx_y, jx_x)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda y, x: residualise(y, x, method='cholesky'),
        ),
        'numpy.linalg.lstsq': ('numpy', _np_residualise),
        'nilearn.signal_clean': (  # canonical fMRI confound-regression floor
            'nilearn', lambda y, x: nilearn_clean(y, np.asarray(x).T)),
        'cupy.linalg.lstsq': ('cupy', _cupy_residualise),  # GPU ref (cuSOLVER)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# fMRI shapes: 400 TRs, 24 confounds, voxel counts the audit used.
_VOXELS = [1000, 10000, 100000]
_N, _K = 400, 24

CASE = Case(
    name='residualise',
    op_qualname='nitrix.linalg.residualise',
    # out[v, :] depends only on Y[v, :] and (all of) X -> a bounded input
    # subset per voxel; oracle computed in full regardless (annex §C).
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'N': _N, 'K': _K, 'seed': 0} for v in _VOXELS],
    # mid voxel count: a large nitrix-vs-lstsq gap without the 100k point's
    # ~160 MB operands slowing the representative / --quick path.
    representative={'V': 10000, 'N': _N, 'K': _K, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
