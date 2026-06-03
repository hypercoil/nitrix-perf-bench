# -*- coding: utf-8 -*-
"""Shared helpers for the spectral-embedding case (laplacian_eigenmap).

``nitrix.graph.laplacian_eigenmap`` returns the smallest **nontrivial**
eigenvalues/eigenvectors of the normalised Laplacian
``L_sym = I - D^-1/2 A D^-1/2`` (the Laplacian-eigenmaps / spectral embedding).
The op exposes two solvers, both benchmarked:

- ``solver='eigh'`` -- dense ``jnp.linalg.eigh`` via nitrix's ``safe_eigh``,
  which on this cuSOLVER-broken L4 **silently runs on CPU** (no hang, but not
  GPU-pure); exact, full-spectrum, fast for small dense graphs;
- ``solver='lobpcg'`` -- matrix-free iterative top-k
  (``jax.experimental.sparse.linalg.lobpcg_standard``), which **runs genuinely
  on the GPU** (dodges the dense cuSOLVER path), the win at large / sparse
  scale.

Eigenvectors carry a sign / degenerate-subspace ambiguity, so the
**eigenvalues** are the clean fidelity target: the case returns them and
scores against an fp64 numpy oracle. References: ``scipy.sparse.linalg.eigsh``
(the recognised
iterative eigen-solver -- the CPU analogue of lobpcg) + a CuPy
``cupyx.scipy.sparse.linalg.eigsh`` GPU ref. scipy is core; cupy lazy.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

_K = 8  # embedding dimension (number of nontrivial components)


def eigenmap_input(n: int, seed: int = 0, density: float = 0.08) -> np.ndarray:
    '''A connected weighted symmetric adjacency: a sparse-structured random
    graph plus a ring (guarantees a single connected component, so the trivial
    eigenvector is unique and the embedding is well-defined).'''
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    w = w * (rng.uniform(0.0, 1.0, (n, n)) < density)
    w = np.triu(w, 1)
    W = w + w.T
    idx = np.arange(n)
    nxt = (idx + 1) % n
    W[idx, nxt] = np.maximum(W[idx, nxt], 0.5)  # connectivity ring
    W[nxt, idx] = W[idx, nxt]
    np.fill_diagonal(W, 0.0)
    return W.astype(np.float32)


def _lsym(W: np.ndarray, xp: Any = np) -> Any:
    '''Symmetric normalised Laplacian ``I - D^-1/2 A D^-1/2``.'''
    d = W.sum(-1)
    dm = 1.0 / xp.sqrt(d)
    n = W.shape[-1]
    return xp.eye(n, dtype=W.dtype) - (dm[:, None] * W * dm[None, :])


def laplacian_eigenvalues(W: Any, k: int = _K) -> np.ndarray:
    '''The k smallest **nontrivial** eigenvalues of L_sym (fp64 oracle).'''
    lsym = _lsym(np.asarray(W, np.float64))
    return np.sort(np.linalg.eigvalsh(lsym))[1:k + 1]


def scipy_eigsh(k: int = _K) -> Callable[[Any], Any]:
    '''Smallest k nontrivial L_sym eigenvalues via scipy.sparse.linalg.eigsh
    (the recognised iterative eigen-solver; CPU floor).'''
    import scipy.sparse.linalg as sla

    def run(W: Any) -> Any:
        lsym = _lsym(np.asarray(W, np.float64))
        ev = sla.eigsh(lsym, k=k + 1, which='SA', return_eigenvectors=False)
        return np.sort(ev)[1:k + 1]

    return run


def cupy_eigsh(k: int = _K) -> Callable[[Any], Any]:
    '''GPU twin via cupyx.scipy.sparse.linalg.eigsh; cupy lazy.'''

    def run(W: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.sparse.linalg as csla

        lsym = _lsym(W, xp=cp)
        ev = csla.eigsh(lsym, k=k + 1, which='SA', return_eigenvectors=False)
        return cp.sort(ev)[1:k + 1]

    return run
