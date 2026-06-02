# -*- coding: utf-8 -*-
"""Shared helpers for the voxelwise LME cases (``nitrix.stats.lme``).

The benchmark uses a **balanced one-way random-intercept** design (``k`` groups
x ``n`` per group, ``N = k·n`` subjects, shared intercept design ``X`` and
group-indicator random-effect design ``Z``), because for that design the REML
variance components have a **closed form** -- a reliable, vectorised fp64
oracle that needs no iterative solver and no external library.

Verified (2026-06-02): nitrix ``reml_fit`` matches this closed form *exactly*
on the variance components and the fixed effect, while ``statsmodels.MixedLM``
-- the canonical CPU library -- can fail to converge near the
variance-component boundary (small ``sigma_b^2``).  So the **closed form is the
oracle** (truth); ``statsmodels`` is the canonical-but-flaky *baseline* (the
real-world looped-CPU comparison), and the few boundary divergences are a
finding, not a bug in nitrix.

Output convention: every estimator returns ``(V, 3)`` columns
``[beta (intercept), sigma_b^2, sigma_e^2]`` so the fidelity compare and the
ratio are over the same quantity.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def balanced_oneway(
    n_vox: int, k: int, n: int, seed: int = 0,
    sigma_b_sq: float = 1.0, sigma_e_sq: float = 1.0, grand: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Balanced one-way random-intercept data.

    Returns ``(Y (V, N), X (N, 1), Z (N, k), groups (N,))`` -- ``X`` is the
    shared intercept, ``Z`` the group indicators, ``groups`` the integer group
    label per subject (for statsmodels).  ``sigma_b_sq`` is kept comfortably
    away from 0 so the canonical statsmodels baseline mostly converges.'''
    rng = np.random.default_rng(seed)
    big_n = k * n
    groups = np.repeat(np.arange(k), n).astype(np.int64)
    X = np.ones((big_n, 1), np.float32)
    Z = np.eye(k, dtype=np.float32)[groups]
    b = rng.standard_normal((n_vox, k)) * np.sqrt(sigma_b_sq)
    eps = rng.standard_normal((n_vox, big_n)) * np.sqrt(sigma_e_sq)
    Y = (grand + b[:, groups] + eps).astype(np.float32)
    return Y, X, Z, groups


def closed_form_reml(Y: np.ndarray, k: int, n: int) -> np.ndarray:
    '''Closed-form balanced one-way REML -> ``(V, 3)`` fp64
    ``[beta, sigma_b^2, sigma_e^2]``.

    For a balanced one-way random-effects model the REML estimates equal the
    ANOVA estimates: ``sigma_e^2 = MSW``, ``sigma_b^2 = max((MSB-MSW)/n, 0)``,
    ``beta = grand mean`` (exact, no iteration).'''
    n_vox, big_n = Y.shape
    yg = Y.reshape(n_vox, k, n)
    group_mean = yg.mean(2)                       # (V, k)
    grand_mean = Y.mean(1)                         # (V,)
    ms_between = n * ((group_mean - grand_mean[:, None]) ** 2).sum(1) / (k - 1)
    ms_within = ((yg - group_mean[:, :, None]) ** 2).sum((1, 2)) / (big_n - k)
    sigma_e_sq = ms_within
    sigma_b_sq = np.maximum((ms_between - ms_within) / n, 0.0)
    return np.stack([grand_mean, sigma_b_sq, sigma_e_sq], axis=-1)


def flame_input(
    n_vox: int, big_n: int, seed: int = 0,
    sigma_b_sq: float = 1.0, s2: float = 0.3, grand: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''FLAME two-level data: per-voxel, per-subject level-1 effects
    ``beta_subject (V, N)`` with a **constant known** within-variance ``s2``
    (so the single-parameter REML has a closed form -- see
    ``flame_closed_form``), and the shared intercept group design.

    Returns ``(beta_subject (V, N), var_within (V, N), X_group (N, 1))``.'''
    rng = np.random.default_rng(seed)
    x_group = np.ones((big_n, 1), np.float32)
    b = rng.standard_normal((n_vox, big_n)) * np.sqrt(sigma_b_sq)
    e = rng.standard_normal((n_vox, big_n)) * np.sqrt(s2)
    beta_subject = (grand + b + e).astype(np.float32)
    var_within = np.full((n_vox, big_n), s2, np.float32)
    return beta_subject, var_within, x_group


def flame_closed_form(
    beta_subject: np.ndarray, x_group: np.ndarray, s2: float,
) -> np.ndarray:
    '''Closed-form FLAME REML for **constant** within-variance ``s2`` -> ``(V,
    2)`` fp64 ``[gamma, sigma_b^2]``.

    With ``s_i^2 = s2`` the model covariance is ``(sigma_b^2 + s2) I``, so the
    REML reduces to GLS == OLS for ``gamma`` and the residual variance for the
    total: ``sigma_b^2 = max(||resid||^2/(N-p) - s2, 0)`` (exact, no
    iteration).'''
    big_n, p = x_group.shape
    xtx_inv = np.linalg.inv(x_group.T @ x_group)        # (p, p)
    gamma = beta_subject @ x_group @ xtx_inv.T          # (V, p)
    resid = beta_subject - gamma @ x_group.T            # (V, N)
    tau2 = (resid ** 2).sum(1) / (big_n - p)
    sigma_b_sq = np.maximum(tau2 - s2, 0.0)
    return np.stack([gamma[:, 0], sigma_b_sq], axis=-1)


def statsmodels_reml(Y: Any, X: Any, groups: Any) -> np.ndarray:
    '''Looped ``statsmodels.MixedLM`` REML -> ``(V, 3)``
    ``[beta, sigma_b^2, sigma_e^2]``.  statsmodels imported lazily (only this
    baseline's worker needs it -- the base env); convergence warnings are
    silenced (boundary non-convergence is expected, surfaced via fidelity).'''
    import warnings

    import statsmodels.api as sm

    yh = np.asarray(Y, np.float64)
    xh = np.asarray(X, np.float64)
    out = np.empty((yh.shape[0], 3), np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for v in range(yh.shape[0]):
            m = sm.MixedLM(yh[v], xh, groups=groups).fit(reml=True)
            out[v] = (m.fe_params[0],
                      float(np.asarray(m.cov_re)[0, 0]), m.scale)
    return out
