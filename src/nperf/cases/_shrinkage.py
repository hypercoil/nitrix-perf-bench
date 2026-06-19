# -*- coding: utf-8 -*-
"""Shared generators + community baselines for the shrinkage-covariance and
sparse-precision (graphical-LASSO) family -- ``nitrix.stats.connectivity``
(``ledoit_wolf`` / ``oas`` / ``shrunk_covariance`` / ``glasso`` /
``glasso_path`` / ``ebic_score``).

The strong community baseline is **scikit-learn** (``sklearn.covariance``): the
canonical reference implementation of Ledoit-Wolf / OAS analytic shrinkage and
the graphical LASSO, so it serves as BOTH the community competitor and the fp64
oracle (nitrix matches it to ~4e-7 on the shrinkage cov, ~4e-4 on the glasso
precision -- verified).  sklearn returns host numpy arrays and lives in the
base env, so its baselines run as the ``numpy`` framework (lazy import).  There
is no on-device (cupy) twin for these estimators, so the GPU headline is
nitrix-jax (jax-cuda12) vs the sklearn CPU bar.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def shrinkage_data(n: int, p: int, seed: int = 0) -> np.ndarray:
    '''``n`` samples x ``p`` features from a low-rank factor model + noise (a
    few latent factors -> a structured covariance), so analytic shrinkage has
    signal -- the ``p >~ n`` high-dimensional regime where Ledoit-Wolf / OAS
    earn their keep (connectome parcels x subjects).'''
    rng = np.random.default_rng(seed)
    k = max(2, p // 20)
    loadings = rng.standard_normal((p, k))
    factors = rng.standard_normal((n, k))
    noise = rng.standard_normal((n, p))
    return (factors @ loadings.T + noise).astype(np.float32)


def emp_cov(X: np.ndarray) -> np.ndarray:
    '''Empirical covariance (the glasso input ``S``); the SAME matrix is fed to
    both nitrix and sklearn so they solve an identical problem.'''
    return np.cov(np.asarray(X), rowvar=False).astype(np.float32)


def sparse_precision_cov(p: int, seed: int = 0, density: float = 0.1
                         ) -> np.ndarray:
    '''A WELL-CONDITIONED population cov ``S = inv(Theta)`` from a random
    SPARSE SPD precision ``Theta`` -- the canonical graphical-LASSO problem (a
    sparse symmetric off-diagonal pattern + a diagonal lift to guarantee SPD).
    Deterministic + well-conditioned, so sklearn's ``graphical_lasso`` solves
    cleanly (the factor-model ``emp_cov`` is near-singular at ``p > n`` and
    makes sklearn raise "Non SPD"). Returns ``S[p, p]`` float32.'''
    rng = np.random.default_rng(seed)
    mask = rng.random((p, p)) < density
    off = np.triu(rng.standard_normal((p, p)) * mask, 1) * 0.3
    theta = off + off.T
    lift = abs(float(np.linalg.eigvalsh(theta).min())) + 0.5  # -> SPD
    theta += np.eye(p) * lift
    return np.linalg.inv(theta).astype(np.float32)


# -- sklearn baselines (the canonical reference impl; community + oracle) -----
def sk_ledoit_wolf(fp64: bool = False) -> Callable[..., Any]:
    def run(X: Any) -> Any:
        from sklearn.covariance import ledoit_wolf
        x = np.asarray(X, np.float64 if fp64 else np.float32)
        return ledoit_wolf(x)[0]   # (cov, shrinkage) -> cov
    return run


def sk_oas(fp64: bool = False) -> Callable[..., Any]:
    def run(X: Any) -> Any:
        from sklearn.covariance import oas
        x = np.asarray(X, np.float64 if fp64 else np.float32)
        return oas(x)[0]
    return run


# NOTE: sklearn's graphical_lasso (coordinate descent) does NOT fully converge
# on these inputs even at max_iter=1000 -- it plateaus at a small NEGATIVE dual
# gap (~1e-4), so the oracle is itself approximate.  This shows up as a non-
# monotonic gate residual (c=160 ~1.6x while its neighbours c=80/320/640 pass),
# i.e. oracle noise, not a nitrix accuracy trend.  The glasso case gate is set
# loose enough to absorb it (see glasso.py); raising max_iter does not help.
def sk_glasso(lam: float, fp64: bool = False) -> Callable[..., Any]:
    def run(S: Any) -> Any:
        from sklearn.covariance import graphical_lasso
        s = np.asarray(S, np.float64 if fp64 else np.float32)
        return graphical_lasso(s, alpha=float(lam))[1]  # (cov, precision)
    return run


def sk_glasso_path(lambdas: Any, fp64: bool = False) -> Callable[..., Any]:
    def run(S: Any) -> Any:
        from sklearn.covariance import graphical_lasso
        s = np.asarray(S, np.float64 if fp64 else np.float32)
        return np.stack([graphical_lasso(s, alpha=float(la))[1]
                         for la in np.asarray(lambdas)])
    return run


def np_ebic(n: int, gamma: float = 0.5, edge_tol: float = 1e-8
            ) -> Callable[..., Any]:
    '''numpy reimpl of nitrix's EBIC (Foygel & Drton 2010):
    ``n (tr(S Theta) - logdet Theta) + E log n + 4 gamma E log p`` with ``E``
    the off-diagonal edge count (upper-tri |theta| > tol).'''
    def run(theta: Any, S: Any) -> Any:
        th = np.asarray(theta, np.float64)
        s = np.asarray(S, np.float64)
        p = th.shape[0]
        sign, logdet = np.linalg.slogdet(th)
        neg2ll = n * (np.sum(s * th) - sign * logdet)
        edges = np.sum(np.triu(np.abs(th) > edge_tol, k=1))
        return np.float64(neg2ll + edges * np.log(n)
                          + 4.0 * gamma * edges * np.log(p))
    return run
