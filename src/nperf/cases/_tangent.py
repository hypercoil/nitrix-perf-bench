# -*- coding: utf-8 -*-
"""Shared helpers for the tangent-space SPD case (tangent_project_spd).

nitrix's ``tangent_project_spd(X, R)`` is the affine-invariant Riemannian log
map ``log(R^-1/2 X R^-1/2)`` at the reference point R -- *not* the legacy
log-Euclidean ``log(X) - log(R)`` (the nitrix docstring is explicit). This is
**exactly** nilearn's connectome tangent kernel: ``ConnectivityMeasure(
kind='tangent')`` whitens with ``W = R^-1/2`` then returns ``logm(W X W)`` via
``connectivity_matrices._map_eigenvalues``. So nilearn is a *faithful*
reference for this primitive -- the canonical neuroimaging tool -- provided the
**same reference R** is given to both. nilearn normally derives R as the
geometric mean of the batch; that geometric-mean iteration is a *separate* op,
deliberately not folded into this primitive benchmark.

The reference R here is a fixed, well-conditioned SPD (the batch Euclidean mean
+ I): the projection kernel is reference-agnostic, so any well-conditioned R
exercises the identical eigh-based code path.

GPU note: like symlog/symsqrt/sympower, both matrix functions (sympower then
symlog) *consume* their eigh into ``V diag(f) Vt``, which XLA lowers to a path
that runs on the GPU on this box, even when the dense potrf/eigh path does not
(a cuSOLVER-class issue observed here -- cause/scope uncharacterised; see
``cases/_spd.py``).
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from ._spd import eig_matrix_fn


def tangent_input(b: int, d: int, seed: int = 0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    '''A batch of ``b`` well-conditioned SPD matrices (d x d) + one SPD
    reference R (the batch Euclidean mean + I -- itself SPD and well
    conditioned). fp32: the realistic GPU workload.'''
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(b):
        m = rng.standard_normal((d, d)).astype(np.float32)
        mats.append((m @ m.T / d + np.eye(d)).astype(np.float32))
    Xs = np.stack(mats)
    R = (Xs.mean(0) + np.eye(d, dtype=np.float32)).astype(np.float32)
    return Xs, R


def eig_tangent(Xs: np.ndarray, R: np.ndarray) -> np.ndarray:
    '''fp64 eigh-based oracle: ``log(R^-1/2 X R^-1/2)`` per matrix.'''
    whiten = eig_matrix_fn(R.astype(np.float64), lambda x: 1.0 / np.sqrt(x))
    return np.stack([
        eig_matrix_fn(whiten @ x.astype(np.float64) @ whiten, np.log)
        for x in Xs
    ])


def nilearn_tangent(Xs: Any, R: Any) -> np.ndarray:
    '''nilearn's connectome tangent kernel (the canonical CPU floor): whiten
    with ``W = R^-1/2`` then matrix-log of ``W X W``, both via nilearn's own
    ``_map_eigenvalues`` -- the exact code path ``ConnectivityMeasure(
    kind='tangent')`` runs. Lazy imports: only the numpy worker needs them.

    BLAS pinned to 1 thread: nilearn's tangent is a Python loop of *small*
    eigh+gemm calls, and multi-threaded OpenBLAS on a small core group thrashes
    its pool on that alternating loop-of-LAPACK pattern (a 15-50x scheduler
    artifact, measured, not nilearn's compute). The eighs are too small to gain
    from threads anyway (1-thread == 2-thread time), so pinning gives nilearn's
    genuine, churn-free cost rather than handicapping it.'''
    from nilearn.connectome.connectivity_matrices import _map_eigenvalues
    from threadpoolctl import threadpool_limits

    R = np.asarray(R)
    with threadpool_limits(limits=1):
        whiten = _map_eigenvalues(lambda x: 1.0 / np.sqrt(x), R)
        out = [_map_eigenvalues(np.log, whiten @ np.asarray(x) @ whiten)
               for x in Xs]
    return np.stack(out)


def cupy_tangent() -> Callable[[Any, Any], Any]:
    '''CuPy eigh-based tangent (the GPU reference); cupy lazy so only the
    refs-cupy worker imports it.'''

    def run(Xs: Any, R: Any) -> Any:
        import cupy as cp

        w, v = cp.linalg.eigh(R)
        whiten = (v * (1.0 / cp.sqrt(w))) @ v.T        # R^-1/2
        wxw = whiten @ Xs @ whiten                     # broadcast -> (b,d,d)
        w2, v2 = cp.linalg.eigh(wxw)                   # batched eigh
        # contiguous transpose: a swapaxes *view* fed to a batched gemm trips
        # CUBLAS_STATUS_INVALID_VALUE on a cold handle (the harness isolates
        # each attempt, so every call is cold).
        vt = cp.ascontiguousarray(cp.swapaxes(v2, -1, -2))
        return cp.matmul(v2 * cp.log(w2)[..., None, :], vt)

    return run
