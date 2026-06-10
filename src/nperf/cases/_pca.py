# -*- coding: utf-8 -*-
"""Shared construction + warranted oracles for the PCA family.

Three nitrix ops share this helper:

* ``pca_fit`` -- the eigh-bound *fit* (covariance eigendecomposition).  Its
  scored quantity is ``explained_variance`` (the top-``k`` covariance
  eigenvalues), **not** the ``components``: principal axes carry a sign (+/-)
  *and* a within-eigenspace rotation ambiguity, so a direct component
  comparison is ill-posed at scale (a degenerate / near-degenerate pair flips
  or rotates under fp32 round-off).  The eigenVALUES are unique and
  rotation/sign-invariant -- the robust fidelity guard.  Basis-application
  correctness is covered by the ``pca_transform`` / ``pca_inverse_transform``
  cases (a fixed shared basis, no ambiguity).

* ``pca_transform`` / ``pca_inverse_transform`` -- pure matmuls against a
  **fixed pre-fitted basis** (``components``, ``mean``) computed once here, so
  every framework consumes the *same* basis and the output is unambiguous.

The data is planted with ``k`` strong directions (geometric eigenvalues, all
O(1)) clearly above an isotropic noise floor, so the top-``k`` variances are
well-conditioned (fp32 eigh matches the fp64 oracle tightly) and the ``k``-th
spectral gap is clean.

GPU note (re-measured 2026-06; see [[perfbench-gpu-eigh-blocker]]):
``pca_fit``'s ``solver='full'`` calls ``safe_eigh`` on the ``(d, d)``
covariance.  The older "routes to CPU at ``d>=256``" assumption did **not**
reproduce in fresh per-attempt workers -- there the cuSOLVER eigh initialises
and runs **GPU-native through d=2048** (no host round-trip).  ``safe_eigh``'s
CPU fallback is a latent net that fires only on a cuSOLVER handle failure,
which here surfaced only in long-lived / memory-pressured contexts (a reused
REPL), not in the workers.  ``pca_transform`` / ``pca_inverse_transform`` are
pure BLAS and never touch a solver.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def pca_input(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    '''An ``(n, d)`` fp32 matrix with a planted rank-``k``-dominant covariance.

    ``k`` orthonormal loading directions carry geometric variances (stdevs
    ``8 -> 1``, all O(1)); an isotropic ``0.05`` noise floor sits two orders of
    magnitude below the weakest signal, so the top-``k`` eigenvalues are
    well-separated from the bulk (a clean ``k``-th gap), well-conditioned.'''
    rng = np.random.default_rng(seed)
    kk = min(k, d, n)
    loadings = rng.standard_normal((d, kk))
    q, _ = np.linalg.qr(loadings)  # (d, kk) orthonormal columns
    scales = np.geomspace(8.0, 1.0, kk)  # signal stdevs, all O(1)
    scores = rng.standard_normal((n, kk)) * scales  # (n, kk)
    signal = scores @ q.T  # (n, d)
    noise = 0.05 * rng.standard_normal((n, d))  # isotropic floor << signal
    return (signal + noise).astype(np.float32)


def np_explained_variance(x64: np.ndarray, k: int) -> np.ndarray:
    '''fp64 oracle for ``pca_fit``: the top-``k`` covariance eigenvalues.

    Mirrors nitrix's ``_pca_full`` (eigh of ``Xc^T Xc / (n - 1)``, descending),
    in double on the same values -- so the gate measures fp32 round-off.'''
    n = x64.shape[0]
    xc = x64 - x64.mean(axis=0)
    cov = (xc.T @ xc) / max(n - 1, 1)
    w = np.linalg.eigvalsh(cov)  # ascending
    return np.sort(w)[::-1][:k]


def np_basis(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    '''A fixed pre-fitted basis ``(components (k, d), mean (d,))`` in fp32.

    Computed once via the fp64 covariance eigh and cast to fp32, then shared
    verbatim by every framework's transform / inverse_transform -- so those
    cases carry *no* sign ambiguity (all consume the identical basis).'''
    n = x.shape[0]
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=0)
    xc = x64 - mean
    cov = (xc.T @ xc) / max(n - 1, 1)
    w, v = np.linalg.eigh(cov)  # ascending
    order = np.argsort(w)[::-1][:k]
    components = v[:, order].T  # (k, d)
    return components.astype(np.float32), mean.astype(np.float32)


def np_transform(
    x: np.ndarray, components: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    '''numpy ``pca_transform``: project on the basis, ``(X - mean) @ C^T``.'''
    return (x - mean) @ components.T


def np_inverse(
    z: np.ndarray, components: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    '''numpy ``pca_inverse_transform``: reconstruct -- ``Z @ C + mean``.'''
    return z @ components + mean


# -- CuPy GPU references (lazy import; only the refs-cupy worker needs cupy) --


def cupy_explained_variance(k: int) -> Any:
    '''CuPy GPU twin of ``pca_fit``'s spectrum: eigh of the ``(d, d)`` cov on
    device (mirrors nitrix's ``solver='full'``), top-``k`` eigenvalues.'''

    def _fn(x: Any) -> Any:
        import cupy as cp

        n = x.shape[0]
        xc = x - x.mean(axis=0)
        cov = (xc.T @ xc) / max(n - 1, 1)
        w = cp.linalg.eigvalsh(cov)
        return cp.sort(w)[::-1][:k]

    return _fn


def cupy_transform(x: Any, components: Any, mean: Any) -> Any:
    '''CuPy GPU ``pca_transform`` -- ``(X - mean) @ C^T`` on device.'''
    return (x - mean) @ components.T


def cupy_inverse(z: Any, components: Any, mean: Any) -> Any:
    '''CuPy GPU ``pca_inverse_transform`` -- ``Z @ C + mean`` on device.'''
    return z @ components + mean


# -- scikit-learn CPU task-level floor for pca_fit ----------------------------


def sklearn_explained_variance(k: int) -> Any:
    '''scikit-learn ``PCA`` (the canonical CPU PCA) -- exact full SVD solver,
    its ``explained_variance_`` is ``S^2 / (n - 1)`` = the same covariance
    eigenvalues nitrix reports.  ``svd_solver='full'`` forces the *exact* path
    (sklearn auto-picks the approximate randomised solver on large inputs).'''

    def _fn(x: Any) -> Any:
        from sklearn.decomposition import PCA

        model = PCA(n_components=k, svd_solver='full').fit(x)
        return model.explained_variance_

    return _fn
