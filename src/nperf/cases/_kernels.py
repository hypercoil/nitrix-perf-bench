# -*- coding: utf-8 -*-
"""Shared helpers for the kernel family (rbf / linear / linear_distance).

``nitrix.linalg.rbf_kernel`` / ``linear_kernel`` / ``linear_distance`` are the
pairwise maps on a feature matrix ``X`` (``n`` rows x ``d`` features) producing
an ``n x n`` Gram / distance matrix: nitrix (jax) vs the canonical CPU kernel
library ``sklearn.metrics.pairwise`` (the recognised reference -- a hand-rolled
numpy floor would invite the "naive baseline" critique) + a CuPy GPU reference,
scored against an fp64 oracle.

Being matmul / broadcast (no eigh / solver), these run **GPU-pure** -- so,
unlike the eigh family, they give a clean apples-to-apples GPU bar at every
size.  CuPy has no ``sklearn.metrics.pairwise``, so the GPU reference computes
the same maps from the Gram matrix directly (``X @ Xᵀ`` and the
``‖x‖² + ‖y‖² - 2⟨x,y⟩`` squared-distance identity, clipped at 0).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ._base import BuiltPoint, Case, to_cupy


def kernel_input(n: int, d: int, seed: int = 0) -> np.ndarray:
    '''A feature matrix: ``n`` samples x ``d`` features (standard normal).'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def cupy_kernel(kind: str, gamma: Optional[float] = None,
                order: int = 3, r: float = 0.0) -> Callable[[Any], Any]:
    '''Build the CuPy GPU kernel baseline; cupy imported lazily so only the
    cupy worker (refs-cupy env) needs it.  ``kind``: ``linear`` (``X @ Xᵀ``) |
    ``distance`` (squared L2 via the Gram identity, clipped) | ``rbf`` /
    ``gaussian`` (``exp(-gamma · ‖x - y‖²)``) | ``cosine`` (Gram / outer norms)
    | ``polynomial`` (``(gamma·G + r)^order``) | ``sigmoid``
    (``tanh(gamma·G + r)``).  CuPy lacks sklearn's pairwise kernels, so we
    derive them from the Gram matrix -- the same maps nitrix computes.'''

    def run(x: Any) -> Any:
        import cupy as cp

        g = x @ x.T
        if kind == 'linear':
            return g
        if kind == 'polynomial':
            return (gamma * g + r) ** order
        if kind == 'sigmoid':
            return cp.tanh(gamma * g + r)
        if kind == 'cosine':
            nrm = cp.sqrt(cp.diagonal(g))
            return g / (nrm[:, None] * nrm[None, :])
        sq = cp.diagonal(g)
        d2 = cp.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
        if kind == 'distance':
            return d2
        return cp.exp(-gamma * d2)  # rbf / gaussian

    return run


def sklearn_kernel(kind: str, gamma: Optional[float] = None,
                   order: int = 3, r: float = 0.0) -> Callable[[Any], Any]:
    '''The canonical ``sklearn.metrics.pairwise`` kernel (CPU floor; fp64
    oracle when fed an fp64 matrix).  Parameter mapping (see kernel.py):
    order->degree, r->coef0; gaussian's sigma folds into gamma=1/(2·sigma^2).
    sklearn lazy-imported (numpy worker only).'''

    def run(x: Any) -> Any:
        from sklearn.metrics import pairwise as P

        xa = np.asarray(x)
        if kind == 'cosine':
            return P.cosine_similarity(xa)
        if kind == 'polynomial':
            return P.polynomial_kernel(xa, degree=order, gamma=gamma, coef0=r)
        if kind == 'sigmoid':
            return P.sigmoid_kernel(xa, gamma=gamma, coef0=r)
        return P.rbf_kernel(xa, gamma=gamma)  # rbf / gaussian

    return run


# sklearn function name per kind (gaussian uses rbf_kernel) -- for the baseline
# label, so the row names the actual reference function.
_SKL_NAME = {'gaussian': 'rbf_kernel', 'cosine': 'cosine_similarity',
             'polynomial': 'polynomial_kernel', 'sigmoid': 'sigmoid_kernel'}
_KERNEL_SHAPES = [(512, 64), (2048, 64), (4096, 64)]


def build_kernel_point(nitrix_call: Callable[[Any], Any], kind: str,
                       param: Dict[str, Any], *, gamma: Optional[float] = None,
                       order: int = 3, r: float = 0.0) -> BuiltPoint:
    '''Shared BuiltPoint for a pairwise-kernel case: the nitrix op vs sklearn
    (CPU floor + fp64 oracle) + cupy (GPU ref), all with matched parameters.'''
    n, d = param['n'], param['d']
    x = kernel_input(n, d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(x))
    skl = sklearn_kernel(kind, gamma=gamma, order=order, r=r)
    ref = skl(x.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Any:
        if framework == 'cupy':
            return to_cupy(x)
        return (x,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', nitrix_call),
        f'sklearn.{_SKL_NAME[kind]}': ('sklearn', skl),
        f'cupy.{kind}_kernel': (
            'cupy', cupy_kernel(kind, gamma=gamma, order=order, r=r)),
    }
    return BuiltPoint(baselines=baselines, inputs_for=inputs_for,
                      fp64_reference=ref, ratio_reference='nitrix-jax')


def kernel_case(name: str, build_fn: Callable[[Dict[str, Any]], BuiltPoint],
                *, atol: float = 1e-4) -> Case:
    '''A pairwise-kernel ``Case`` over the shared feature-matrix shapes.'''
    return Case(
        name=name, op_qualname=f'nitrix.linalg.{name}',
        output_independent=False,  # K[i,j] couples rows i and j
        metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
                 'throughput'],
        param_points=[{'n': n, 'd': d, 'seed': 0}
                      for (n, d) in _KERNEL_SHAPES],
        representative={'n': 2048, 'd': 64, 'seed': 0},
        build=build_fn, rtol=1e-3, atol=atol,
    )
