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

from typing import Any, Callable, Optional

import numpy as np


def kernel_input(n: int, d: int, seed: int = 0) -> np.ndarray:
    '''A feature matrix: ``n`` samples x ``d`` features (standard normal).'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def cupy_kernel(kind: str, gamma: Optional[float] = None
                ) -> Callable[[Any], Any]:
    '''Build the CuPy GPU kernel baseline; cupy imported lazily so only the
    cupy worker (refs-cupy env) needs it.  ``kind``: ``linear`` (``X @ Xᵀ``) |
    ``distance`` (squared L2 via the Gram identity, clipped) | ``rbf``
    (``exp(-gamma · ‖x - y‖²)``).  CuPy lacks sklearn's pairwise kernels, so we
    derive them from the Gram matrix -- the same maps nitrix computes.'''

    def run(x: Any) -> Any:
        import cupy as cp

        g = x @ x.T
        if kind == 'linear':
            return g
        sq = cp.diagonal(g)
        d2 = cp.maximum(sq[:, None] + sq[None, :] - 2.0 * g, 0.0)
        if kind == 'distance':
            return d2
        return cp.exp(-gamma * d2)  # rbf

    return run
