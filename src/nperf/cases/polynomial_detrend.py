# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.signal.polynomial_detrend`` vs numpy / cupy.

Subtract a degree-``d`` polynomial fit from each channel (== ``residualise``
against a Vandermonde basis): nitrix (jax) vs a numpy least-squares detrend
(the CPU floor) + a CuPy least-squares detrend (GPU ref), scored against an
fp64 Vandermonde-residual oracle.  The residual is invariant to the basis
scaling (it is the projection onto the orthogonal complement of the degree-d
polynomials), so any polynomial basis of the same degree -- nitrix's rescaled
Vandermonde, ours -- yields the same answer.  The fit is a tiny tall-skinny
lstsq (``obs x (d+1)``, d+1 <= 4), so it is GPU-pure (no cuSolver blocker): a
clean apples-to-apples GPU bar.  Ratio vs the numpy detrend.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.signal import polynomial_detrend

from ._base import BuiltPoint, Case, to_cupy

_DEGREE = 3


def _np_detrend(x: np.ndarray, degree: int) -> np.ndarray:
    '''Residual of ``x`` (``C x obs``) after removing a degree-``degree``
    polynomial fit per channel, via a Vandermonde least-squares fit.'''
    obs = x.shape[-1]
    t = np.linspace(-1.0, 1.0, obs, dtype=x.dtype)
    V = np.vander(t, degree + 1, increasing=True)  # (obs, degree+1)
    coef, *_ = np.linalg.lstsq(V, x.T, rcond=None)  # (degree+1, C)
    return x - (V @ coef).T


def _cupy_detrend(x: Any, degree: int) -> Any:
    '''GPU twin of ``_np_detrend`` (cupy lazy; refs-cupy env).'''
    import cupy as cp

    obs = x.shape[-1]
    t = cp.linspace(-1.0, 1.0, obs, dtype=x.dtype)
    V = cp.vander(t, degree + 1, increasing=True)
    coef, *_ = cp.linalg.lstsq(V, x.T, rcond=None)
    return x - (V @ coef).T


def _build(param: Dict[str, Any]) -> BuiltPoint:
    c, obs, deg = param['c'], param['obs'], param.get('degree', _DEGREE)
    rng = np.random.default_rng(param.get('seed', 0))
    t = np.linspace(-1.0, 1.0, obs)
    trend = (rng.standard_normal((c, 1)) * t ** 3
             + rng.standard_normal((c, 1)) * t ** 2
             + rng.standard_normal((c, 1)) * t)
    X = (trend + 0.1 * rng.standard_normal((c, obs))).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = _np_detrend(X.astype(np.float64), deg)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: polynomial_detrend(x, degree=deg)),
        'numpy.lstsq_detrend': ('numpy', lambda x: _np_detrend(x, deg)),
        'cupy.lstsq_detrend': ('cupy', lambda x: _cupy_detrend(x, deg)),  # GPU
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, observations): cost ~ C * obs * (degree+1); detrend each channel.
_SHAPES = [(256, 2048), (1024, 4096), (4096, 4096)]

CASE = Case(
    name='polynomial_detrend',
    op_qualname='nitrix.signal.polynomial_detrend',
    output_independent=False,  # each residual depends on the whole-channel fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'obs': o, 'degree': _DEGREE, 'seed': 0}
                  for (c, o) in _SHAPES],
    representative={'c': 1024, 'obs': 4096, 'degree': _DEGREE, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
