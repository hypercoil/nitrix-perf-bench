# -*- coding: utf-8 -*-
"""Shared generators + baselines for the Gaussian-likelihood family
(``nitrix.stats.gaussian_nll`` / ``kl_diagonal_gaussian``) -- the VAE / latent
losses.  Cheap elementwise + a global reduction, so the apples-to-apples GPU
bar is a **cupy** reimplementation (like ``cov``); the fp64 oracle is the exact
numpy formula, and ``scipy.stats.norm`` is a community cross-check for the NLL.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Tuple

import numpy as np

_LOG2PI = math.log(2.0 * math.pi)


def gaussian_inputs(shape, seed: int = 0) -> Tuple[np.ndarray, ...]:
    '''(x, mean, log_var) at ``shape`` -- a predicted-Gaussian regime: data x,
    predicted mean, and a modest log-variance (kept in [-2, 2] so exp(log_var)
    is well-scaled).'''
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    mean = rng.standard_normal(shape).astype(np.float32)
    log_var = (rng.standard_normal(shape) * 0.7).clip(-2, 2).astype(np.float32)
    return x, mean, log_var


def np_gaussian_nll(fp64: bool = False) -> Callable[..., Any]:
    dt = np.float64 if fp64 else np.float32
    def run(x: Any, mean: Any, log_var: Any) -> Any:
        x = np.asarray(x, dt)
        m = np.asarray(mean, dt)
        lv = np.asarray(log_var, dt)
        elem = 0.5 * (_LOG2PI + lv + (x - m) ** 2 / np.exp(lv))
        return elem.mean()           # reduction='mean'
    return run


def np_kl_diag(fp64: bool = False) -> Callable[..., Any]:
    dt = np.float64 if fp64 else np.float32
    def run(mean: Any, log_var: Any) -> Any:
        m = np.asarray(mean, dt)
        lv = np.asarray(log_var, dt)
        return 0.5 * np.sum(np.exp(lv) + m ** 2 - 1.0 - lv)  # reduction='sum'
    return run


def scipy_gaussian_nll() -> Callable[..., Any]:
    '''-mean(norm.logpdf(x; mean, exp(log_var/2))) -- the same NLL via
    scipy.stats (the community cross-check).'''
    def run(x: Any, mean: Any, log_var: Any) -> Any:
        from scipy.stats import norm
        x = np.asarray(x, np.float32)
        m = np.asarray(mean, np.float32)
        sd = np.exp(0.5 * np.asarray(log_var, np.float32))
        return -norm.logpdf(x, loc=m, scale=sd).mean()
    return run


def cupy_gaussian_nll() -> Callable[..., Any]:
    def run(x: Any, mean: Any, log_var: Any) -> Any:
        import cupy as cp
        elem = 0.5 * (_LOG2PI + log_var + (x - mean) ** 2 / cp.exp(log_var))
        return elem.mean()
    return run


def cupy_kl_diag() -> Callable[..., Any]:
    def run(mean: Any, log_var: Any) -> Any:
        import cupy as cp
        return 0.5 * cp.sum(cp.exp(log_var) + mean ** 2 - 1.0 - log_var)
    return run
