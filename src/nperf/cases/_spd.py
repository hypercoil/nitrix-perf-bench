# -*- coding: utf-8 -*-
"""Shared helpers for the SPD eigh-family cases (symlog / symsqrt / sympower).

These ops are symmetric **matrix functions** ``f(A) = V diag(f(λ)) Vᵀ`` on an
SPD ``A`` (nitrix computes them via ``eigh``).  The fp64 oracle and the CuPy
GPU reference both compute ``f`` via an explicit symmetric eigendecomposition
(numpy / cupy), mirroring nitrix's own method -- so for **well-conditioned**
SPD inputs nitrix-jax, scipy, cupy, and the oracle all agree to round-off.

**GPU note (cuSOLVER eigh broken at d≥256; matrix functions dodge it).**  Dense
cuSOLVER ``eigh`` is broken at d≥256 on this L4 / driver-580 stack: both
``cupy.linalg.eigh`` (standalone) and a **bare** ``jnp.linalg.eigh`` (eager and
jitted) fail with a cuSolver internal error; d=64 works.  But these are
**matrix functions** that *consume* the decomposition into ``V diag(f(λ)) Vᵀ``,
and (tested) XLA lowers a *consumed* jitted eigh to a non-cuSolver path -- so
**nitrix runs symlog/symsqrt/sympower on the GPU honestly** (verified correct
in a provably cuda-only process), where cupy / bare eigh cannot.  This is *not*
``safe_eigh`` (a raw-eigh matrix-log behaves identically) and does **not**
extend to ops that *return* eigenpairs (e.g. ``laplacian_eigenmap``, which
would still hit the cuSolver failure).  The trade-off: the **cupy GPU ref fails
at d≥256** (recorded ``gpu_solver_unavailable``), so the apples-to-apples GPU
bar holds only at d=64.  ``scipy.linalg`` is the CPU floor.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def spd_input(d: int, seed: int = 0) -> np.ndarray:
    '''A well-conditioned SPD matrix (eigenvalues ≳ 1, so the matrix functions
    are numerically benign and all methods agree).'''
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((d, d)).astype(np.float32)
    return (m @ m.T / d + np.eye(d)).astype(np.float32)


def eig_matrix_fn(a: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]
                  ) -> np.ndarray:
    '''``f(A) = V·diag(fn(λ))·Vᵀ`` for symmetric ``A`` via numpy ``eigh`` --
    the fp64 oracle (and the unambiguous SPD matrix-function definition).'''
    w, v = np.linalg.eigh(a)
    return (v * fn(w)) @ v.T


def cupy_matrix_fn(kind: str, power: float = 1.0) -> Callable[[Any], Any]:
    '''Build the CuPy eigh-based matrix-function baseline (GPU); cupy lazy so
    only the cupy worker (refs-cupy env) imports it.  CuPy's eigh works on the
    L4 where jax's does not (the jaxlib cuSOLVER blocker).'''

    def run(a: Any) -> Any:
        import cupy as cp

        w, v = cp.linalg.eigh(a)
        if kind == 'log':
            fw = cp.log(w)
        elif kind == 'sqrt':
            fw = cp.sqrt(w)
        elif kind == 'exp':
            fw = cp.exp(w)
        else:
            fw = cp.power(w, power)
        return (v * fw) @ v.T

    return run
