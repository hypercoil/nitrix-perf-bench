# -*- coding: utf-8 -*-
"""Shared helpers for the graph family (laplacian / modularity_matrix).

Both ops take a dense weighted adjacency ``A`` (n x n, symmetric, zero
diagonal) and are pure matmul/broadcast -> GPU-pure (no solver).

References (verified exact in fp64):
- ``laplacian`` (combinatorial ``D - A``) == ``scipy.sparse.csgraph.laplacian(
  normed=False)`` -- the array-based scientific standard (fair kernel-vs-kernel
  CPU floor);
- ``modularity_matrix`` (``A - k kᵀ / 2m``) == ``networkx.modularity_matrix(G,
  weight='weight')`` -- the canonical graph library (the *weighted* degree/null
  term; the default ``weight=None`` uses the binary adjacency and does NOT
  match). networkx is graph-object-based, so the floor includes graph
  construction from the array (the honest end-to-end networkx cost).

scipy.sparse.csgraph is a core dep; networkx / cupy are lazy (their workers
only).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def graph_input(n: int, seed: int = 0, density: float = 0.15) -> np.ndarray:
    '''A random weighted, symmetric, zero-diagonal adjacency (sparse-structured
    dense array); ``density`` is the fraction of present edges.'''
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 1.0, (n, n)).astype(np.float32)
    w = w * (rng.uniform(0.0, 1.0, (n, n)) < density)
    w = np.triu(w, 1)
    return (w + w.T).astype(np.float32)


def scipy_laplacian() -> Callable[[Any], Any]:
    '''Combinatorial Laplacian ``D - A`` via scipy.sparse.csgraph (CPU floor +
    fp64 oracle); array-based, the scientific standard.'''
    import scipy.sparse.csgraph as csg

    def run(a: Any) -> Any:
        return csg.laplacian(np.asarray(a), normed=False)

    return run


def np_modularity(a: Any) -> np.ndarray:
    '''Newman modularity matrix ``A - k kᵀ/2m`` in numpy (fp64 oracle).'''
    a = np.asarray(a)
    k = a.sum(-1)
    m = a.sum() / 2.0
    return a - np.outer(k, k) / (2.0 * m)


def np_degree(a: Any) -> np.ndarray:
    '''Per-node degree (row sum) -- the textbook numpy floor + fp64 oracle.'''
    return np.asarray(a).sum(-1)


def np_gn_null(a: Any) -> np.ndarray:
    '''Girvan-Newman null ``k kᵀ/2m`` in numpy (floor + fp64 oracle).'''
    a = np.asarray(a)
    k = a.sum(-1)
    m = a.sum() / 2.0
    return np.outer(k, k) / (2.0 * m)


def cupy_degree() -> Callable[[Any], Any]:
    '''GPU per-node degree (row sum); operates on the cupy array directly.'''
    return lambda a: a.sum(-1)


def cupy_gn_null() -> Callable[[Any], Any]:
    '''GPU Girvan-Newman null ``k kᵀ / 2m``; cupy lazy.'''

    def run(a: Any) -> Any:
        import cupy as cp

        k = a.sum(-1)
        m = a.sum() / 2.0
        return cp.outer(k, k) / (2.0 * m)

    return run


def nx_modularity() -> Callable[[Any], Any]:
    '''``networkx.modularity_matrix`` (weighted) -- the canonical graph-library
    reference; builds the Graph from the array (honest end-to-end cost).
    networkx lazy so only the numpy worker imports it.'''

    def run(a: Any) -> Any:
        import networkx as nx

        g = nx.from_numpy_array(np.asarray(a))
        return np.asarray(nx.modularity_matrix(g, weight='weight'))

    return run


def cupy_laplacian() -> Callable[[Any], Any]:
    '''GPU combinatorial Laplacian ``diag(rowsum) - A``; cupy lazy.'''

    def run(a: Any) -> Any:
        import cupy as cp

        return cp.diag(a.sum(-1)) - a

    return run


def cupy_modularity() -> Callable[[Any], Any]:
    '''GPU modularity matrix ``A - k kᵀ / 2m``; cupy lazy.'''

    def run(a: Any) -> Any:
        import cupy as cp

        k = a.sum(-1)
        m = a.sum() / 2.0
        return a - cp.outer(k, k) / (2.0 * m)

    return run
