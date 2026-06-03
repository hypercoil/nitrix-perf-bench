# -*- coding: utf-8 -*-
"""Shared helpers for the diffusion-map case (diffusion_embedding).

``nitrix.graph.diffusion_embedding`` is the Coifman-Lafon diffusion map: the
**largest** nontrivial eigenpairs of the anisotropic diffusion operator. With
``alpha`` density-normalisation, the symmetric conjugate of the Markov operator
is ``P_sym = D_a^-1/2 A_a D_a^-1/2`` where ``A_a = D^-alpha A D^-alpha`` -- its
eigenvalues are the diffusion eigenvalues (verified to match nitrix to ~7e-8).

Like ``laplacian_eigenmap`` it exposes the two solvers (eigh via safe_eigh ->
CPU on the broken cuSOLVER stack; lobpcg matrix-free -> genuine GPU), and the
eigenvalues are the clean fidelity target (eigenvectors carry sign ambiguity).
There is no standard diffusion-map library, so the reference is the
well-defined operator's own ``scipy.sparse.linalg.eigsh`` (largest) + a CuPy
eigsh GPU ref,
scored against an fp64 numpy oracle. scipy is core; cupy lazy.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._eigenmap import _K, eigenmap_input  # noqa: F401  (re-export the input)

_ALPHA = 0.5  # Coifman-Lafon density normalisation (Fokker-Planck)


def _psym(W: Any, alpha: float, xp: Any = np) -> Any:
    '''Symmetric conjugate of the anisotropic diffusion (Markov) operator.'''
    d = W.sum(-1)
    aa = (d ** -alpha)[:, None] * W * (d ** -alpha)[None, :]
    da = aa.sum(-1)
    return (da ** -0.5)[:, None] * aa * (da ** -0.5)[None, :]


def diffusion_eigenvalues(W: Any, k: int = _K,
                          alpha: float = _ALPHA) -> np.ndarray:
    '''The k largest **nontrivial** diffusion eigenvalues (fp64 oracle).'''
    psym = _psym(np.asarray(W, np.float64), alpha)
    return np.sort(np.linalg.eigvalsh(psym))[::-1][1:k + 1]


def scipy_eigsh_diffusion(k: int = _K,
                          alpha: float = _ALPHA) -> Callable[[Any], Any]:
    '''Largest k nontrivial diffusion eigenvalues, scipy eigsh (CPU floor).'''
    import scipy.sparse.linalg as sla

    def run(W: Any) -> Any:
        psym = _psym(np.asarray(W, np.float64), alpha)
        ev = sla.eigsh(psym, k=k + 1, which='LA', return_eigenvectors=False)
        return np.sort(ev)[::-1][1:k + 1]

    return run


def cupy_eigsh_diffusion(k: int = _K,
                         alpha: float = _ALPHA) -> Callable[[Any], Any]:
    '''GPU twin via cupyx eigsh (largest); cupy lazy.'''

    def run(W: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.sparse.linalg as csla

        psym = _psym(W, alpha, xp=cp)
        ev = csla.eigsh(psym, k=k + 1, which='LA', return_eigenvectors=False)
        return cp.sort(ev)[::-1][1:k + 1]

    return run
