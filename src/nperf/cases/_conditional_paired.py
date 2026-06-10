# -*- coding: utf-8 -*-
"""Shared construction + warranted oracles for the paired / conditional family.

Four nitrix ops, two pairs, built from a shared latent-factor input so each is
a *meaningful* test (the cross-block and the conditioning both carry real
signal):

* ``pairedcov(X, Y)`` / ``pairedcorr(X, Y)`` -- the **cross**-covariance /
  -correlation between two variable blocks, ``Xc @ Yc^T / (obs - 1)`` shaped
  ``(c, d)`` (and, for corr, divided by the geometric mean of the two blocks'
  per-variable variances).  Pure BLAS -- no solver.

* ``conditionalcov(X, Y)`` / ``conditionalcorr(X, Y)`` -- the covariance /
  correlation of ``X`` **after residualising out** the ``Y`` subspace (OLS):
  ``cov(X - Y-projection)``, shaped ``(c, c)``.  The only solver is the
  normal-equations Cholesky on the **tiny ``(d, d)`` confound Gram** (``d`` the
  number of conditioning variables -- a handful of nuisance regressors in the
  realistic fMRI framing), so unlike ``pca_fit``'s ``(d, d)`` eigh at parcel
  ``d`` these stay matmul-dominated and GPU-robust.

The reference replicates nitrix's **exact** conventions (verified to ~1e-16 vs
the jitted op): ``ddof = 1`` (``bias=False`` default), ``rowvar=True`` (rows
are variables), the residualisation regresses on the **raw** ``Y`` columns (no
intercept -- the subsequent ``cov`` centres), and the corr normalisations match
``_corrnorm`` / ``pairedcorr`` (``+eps`` *outside* the sqrt-of-product).

Why no nilearn floor here (contrast the precision family's
``ConnectivityMeasure``): nilearn maps ``precision`` / ``partial correlation``
directly, but it has **no** cross- or conditional-covariance kind.
``nilearn.signal.clean(confounds=Y)`` residualises, but it adds an intercept
and (by default) detrends / standardises -- a *different* estimator -- so
forcing it here would compare apples to oranges.  numpy is the exact CPU floor;
cupy is the on-device GPU twin.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def paired_input(
    c: int, d: int, obs: int, seed: int = 0, rank: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    '''Two variable x observation blocks ``X (c, obs)``, ``Y (d, obs)`` sharing
    ``rank`` latent factors.

    The shared factors give a genuine **cross**-covariance (so paired-cov/corr
    are non-trivial) *and* make ``X`` partly predictable from ``Y`` (so
    residualising ``Y`` out actually moves the conditional covariance away from
    plain ``cov(X)`` -- a meaningful conditioning test).  An isotropic noise
    floor keeps ``Y`` full row-rank (so the OLS Gram is non-singular) and ``X``
    not perfectly explained.'''
    rng = np.random.default_rng(seed)
    r = min(rank, c, d, obs)
    Z = rng.standard_normal((r, obs))  # shared latent factors
    Lx = rng.standard_normal((c, r))
    Ly = rng.standard_normal((d, r))
    X = Lx @ Z + 0.5 * rng.standard_normal((c, obs))
    Y = Ly @ Z + 0.5 * rng.standard_normal((d, obs))
    return X.astype(np.float32), Y.astype(np.float32)


def paired_conditional(X: Any, Y: Any, kind: str, xp: Any) -> Any:
    '''Compute one of the four members of ``X``, ``Y`` via array module ``xp``
    (``numpy`` or ``cupy``), in ``X``'s dtype -- nitrix's exact convention
    (``ddof=1``, ``rowvar=True``, no-intercept residualisation).'''
    n = X.shape[-1]
    Xc = X - X.mean(axis=-1, keepdims=True)
    if kind in ('pairedcov', 'pairedcorr'):
        Yc = Y - Y.mean(axis=-1, keepdims=True)
        sigma_xy = Xc @ Yc.T / (n - 1)  # (c, d) cross-covariance
        if kind == 'pairedcov':
            return sigma_xy
        var_x = (Xc * Xc).sum(axis=-1) / (n - 1)  # diag(cov(X)) -- (c,)
        var_y = (Yc * Yc).sum(axis=-1) / (n - 1)  # diag(cov(Y)) -- (d,)
        norm = xp.sqrt(var_x[:, None] * var_y[None, :])
        return sigma_xy / (norm + xp.finfo(norm.dtype).eps)

    # conditional: OLS-residualise X against the raw Y columns, then cov.
    gram = Y @ Y.T  # (d, d) confound Gram -- tiny
    rhs = Y @ X.T  # (d, c)
    betas = xp.linalg.solve(gram, rhs)  # (d, c); same projection as Cholesky
    resid = X - (Y.T @ betas).T  # (c, obs)
    Rc = resid - resid.mean(axis=-1, keepdims=True)
    sigma = Rc @ Rc.T / (n - 1)  # (c, c)
    if kind == 'conditionalcov':
        return sigma
    diag = xp.sqrt(xp.diagonal(sigma))
    norm = diag[:, None] * diag[None, :] + xp.finfo(diag.dtype).eps
    return sigma / norm


def cupy_paired_conditional(kind: str) -> Callable[[Any, Any], Any]:
    '''CuPy GPU twin for the given member; cupy imported lazily so only the
    refs-cupy worker needs it.'''

    def run(x: Any, y: Any) -> Any:
        import cupy as cp

        return paired_conditional(x, y, kind, cp)

    return run
