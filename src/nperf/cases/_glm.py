# -*- coding: utf-8 -*-
"""Shared generators + community baselines for the mass-univariate GLM family
(``nitrix.stats.glm_fit`` / ``t_contrast`` / ``f_contrast``).

The neuroimaging workload: fit an OLS GLM at every voxel (``Y[V, N]`` = V
voxels x N observations, regressed on a shared design ``X[N, p]``) and read a
contrast.  nitrix vmaps all V voxels behind one fit; the community tools are
**nilearn** (``run_glm`` -- vectorised, the neuro mass-univariate standard) and
**statsmodels** (per-voxel ``OLS`` loop -- the slow reference).  The fp64
oracle is an exact numpy OLS (lstsq betas / closed-form t / F).
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def glm_data(V: int, N: int, p: int, seed: int = 0
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''(Y[V, N], X[N, p]): V voxels x N observations + a shared design whose
    first column is the intercept; Y = X @ beta + noise.'''
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p)).astype(np.float32)
    X[:, 0] = 1.0
    beta = rng.standard_normal((p, V)).astype(np.float32)
    Y = (X @ beta + 0.5 * rng.standard_normal((N, V))).T  # [V, N]
    return Y.astype(np.float32), X


# -- numpy fp64 oracles (exact OLS) ------------------------------------------
def _ols(Y: np.ndarray, X: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)   # [p, V]
    resid = Y.T - X @ beta                                # [N, V]
    dof = X.shape[0] - X.shape[1]
    disp = (resid ** 2).sum(0) / dof                      # [V]
    xtx_inv = np.linalg.inv(X.T @ X)
    return beta, disp, xtx_inv, dof


def np_glm_beta() -> Callable[..., Any]:
    def run(Y: Any, X: Any) -> Any:
        beta, *_ = _ols(np.asarray(Y, np.float64), np.asarray(X, np.float64))
        return beta.T                                     # [V, p] = .coef
    return run


def np_t_stat(c: np.ndarray) -> Callable[..., Any]:
    cc = np.asarray(c, np.float64)
    def run(Y: Any, X: Any) -> Any:
        beta, disp, xtx_inv, _ = _ols(np.asarray(Y, np.float64),
                                      np.asarray(X, np.float64))
        se = np.sqrt(disp * (cc @ xtx_inv @ cc))
        return (cc @ beta) / se                           # t [V]
    return run


def np_f_stat(C: np.ndarray) -> Callable[..., Any]:
    CC = np.asarray(C, np.float64)
    m = CC.shape[0]
    def run(Y: Any, X: Any) -> Any:
        beta, disp, xtx_inv, _ = _ols(np.asarray(Y, np.float64),
                                      np.asarray(X, np.float64))
        cb = CC @ beta                                    # [m, V]
        M = np.linalg.inv(CC @ xtx_inv @ CC.T)
        return np.einsum('iv,ij,jv->v', cb, M, cb) / m / disp
    return run


# -- nilearn (vectorised mass-univariate -- the community standard) ----------
def nilearn_glm_beta() -> Callable[..., Any]:
    def run(Y: Any, X: Any) -> Any:
        from nilearn.glm.first_level import run_glm
        # nilearn Y is [N, V] (the transpose of nitrix's [V, N]).
        labels, est = run_glm(np.asarray(Y, np.float64).T,
                              np.asarray(X, np.float64), noise_model='ols')
        return est[labels[0]].theta.T                     # [V, p]
    return run


def nilearn_contrast(con: np.ndarray, kind: str) -> Callable[..., Any]:
    cv = np.asarray(con, np.float64)
    def run(Y: Any, X: Any) -> Any:
        from nilearn.glm.contrasts import compute_contrast
        from nilearn.glm.first_level import run_glm
        labels, est = run_glm(np.asarray(Y, np.float64).T,
                              np.asarray(X, np.float64), noise_model='ols')
        ct = compute_contrast(labels, est, cv, stat_type=kind)
        return ct.stat()                                  # [V]
    return run


# -- statsmodels (per-voxel OLS loop -- the slow reference) ------------------
def statsmodels_glm_beta() -> Callable[..., Any]:
    def run(Y: Any, X: Any) -> Any:
        import statsmodels.api as sm
        Y = np.asarray(Y, np.float64)
        X = np.asarray(X, np.float64)
        return np.stack([sm.OLS(Y[v], X).fit().params
                         for v in range(Y.shape[0])])      # [V, p]
    return run


def statsmodels_contrast(con: np.ndarray, kind: str) -> Callable[..., Any]:
    cv = np.asarray(con, np.float64)
    def run(Y: Any, X: Any) -> Any:
        import statsmodels.api as sm
        Y = np.asarray(Y, np.float64)
        X = np.asarray(X, np.float64)
        out = []
        for v in range(Y.shape[0]):
            fit = sm.OLS(Y[v], X).fit()
            out.append(float(np.ravel(fit.t_test(cv).tvalue)[0]) if kind == 't'
                       else float(fit.f_test(cv).fvalue))
        return np.asarray(out)                             # [V]
    return run
