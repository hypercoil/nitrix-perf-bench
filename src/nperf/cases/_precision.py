# -*- coding: utf-8 -*-
"""Shared helpers for the precision / partial-covariance family.

``nitrix.stats.{precision, partialcov, partialcorr}`` are all built from the
**inverse covariance** of a variable x observation matrix:

- ``precision``  = ``inv(cov(X))``
- ``partialcov`` = precision with off-diagonals negated (``P · (2I - 1)``)
- ``partialcorr``= ``-P_ij / sqrt(P_ii · P_jj)`` (diagonal 1) -- partialcov
  normalised by the geometric mean of its absolute diagonal.

The reference replicates that **exact construction** (the off-diagonal sign
flip + the ``|diag|`` normalisation are the subtle parts -- a naive ``inv``
reference gets partialcorr signs/scale wrong), and matches nitrix's ``cov``
normalisation (default ``bias=False`` == numpy/cupy ``cov`` default).

**GPU (eigh-family pattern).**  Jitted nitrix runs these on the GPU: XLA lowers
the consumed ``inv`` off the (broken-on-this-L4) cuSolver path -- verified the
ops succeed on GPU even while a bare ``jnp.linalg.eigh`` is wedged (see
[[perfbench-gpu-eigh-blocker]]).  The CuPy reference, by contrast, calls
``cupy.linalg.inv`` -> cuSolver ``getrf``, which fails at large ``c`` (and
whenever the device's cuSolver is wedged) -- so the apples-to-apples GPU bar
holds only at small ``c`` on a healthy device; the large-``c`` cupy failures
are recorded ``gpu_solver_unavailable`` and are themselves evidence of nitrix's
robustness.  ``scipy``/numpy is the CPU floor.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def precision_input(c: int, obs: int, seed: int = 0) -> np.ndarray:
    '''A variable x observation matrix (``c`` variables, ``obs`` samples);
    ``obs > c`` keeps the sample covariance non-singular (``precision`` uses a
    plain ``inv`` by default).'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((c, obs)).astype(np.float32)


def inv_family(X: Any, kind: str, xp: Any) -> Any:
    '''Compute ``precision`` / ``partialcov`` / ``partialcorr`` of ``X`` via
    array module ``xp`` (``numpy`` or ``cupy``), in ``X``'s dtype.

    ``xp.cov`` upcasts to fp64 (numpy/cupy convention), so we cast the
    covariance back to the input precision before the inverse -- a fair,
    same-precision comparison against nitrix (which stays in fp32).'''
    cov = xp.cov(X).astype(X.dtype)
    P = xp.linalg.inv(cov)
    if kind == 'precision':
        return P
    n = P.shape[-1]
    omega = P * (2.0 * xp.eye(n, dtype=P.dtype) - 1.0)  # negate off-diagonal
    if kind == 'partialcov':
        return omega
    d = xp.sqrt(xp.abs(xp.diagonal(omega)))
    norm = d[:, None] * d[None, :] + xp.finfo(d.dtype).eps
    return omega / norm


def cupy_inv_family(kind: str) -> Callable[[Any], Any]:
    '''CuPy GPU baseline for the given member; cupy imported lazily so only the
    cupy worker (refs-cupy env) needs it.'''

    def run(x: Any) -> Any:
        import cupy as cp

        return inv_family(x, kind, cp)

    return run
