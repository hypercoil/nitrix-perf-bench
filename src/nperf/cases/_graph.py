# -*- coding: utf-8 -*-
"""Shared helpers for the graph family.

Adjacency ops (``laplacian`` / ``modularity_matrix`` / ``degree_vector`` /
``girvan_newman_null``) take a dense weighted adjacency ``A`` (n x n,
symmetric, zero diagonal); the community ops (``coaffiliation`` /
``relaxed_modularity``) also take a community-assignment matrix ``C`` (n x k).
All are pure matmul/broadcast/reduction -> GPU-pure (no solver).

References (verified exact in fp64):
- ``laplacian`` (combinatorial ``D - A``) == ``scipy.sparse.csgraph.laplacian(
  normed=False)`` -- the array-based scientific standard (fair kernel-vs-kernel
  CPU floor);
- ``modularity_matrix`` (``A - k kᵀ / 2m``) == ``networkx.modularity_matrix(G,
  weight='weight')`` -- the canonical graph library (the *weighted* degree/null
  term; the default ``weight=None`` uses the binary adjacency and does NOT
  match). networkx is graph-object-based, so the floor includes graph
  construction from the array (the honest end-to-end networkx cost).
- ``relaxed_modularity`` (hard one-hot ``C``, ``exclude_diag=False``) ==
  ``networkx.community.modularity(G, weight='weight') / 2`` -- verified exact
  in fp64 across seeds/sizes/k/gamma. The ``/2`` is a real convention bridge:
  nitrix corrects the undirected double-count *twice* (once via the ``1/2m``
  prefactor, once via an explicit ``Q/2``), so its score is the canonical
  Newman modularity halved (the op docstring's "reduces to the standard Newman
  modularity" is off by this factor; filed low-priority with nitrix). We
  benchmark ``exclude_diag=False`` -- the canonical-comparable mode; the
  default ``exclude_diag=True`` additionally drops the within-community
  diagonal term.
- ``coaffiliation`` (``C Cᵀ`` with zeroed diagonal) has no canonical external
  library -- a nitrix-specific primitive -- so a numpy outer-product floor +
  fp64 oracle + a CuPy GPU ref (the degree_vector / girvan_newman_null model).

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


def assignment_input(n: int, k: int = 16, seed: int = 0) -> np.ndarray:
    '''A random *soft* community-assignment matrix ``C`` (n x k), nonnegative
    (logit-like) -- the general overlapping case that exercises the full
    ``C Cᵀ`` Gram path of ``coaffiliation``.'''
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, (n, k)).astype(np.float32)


def partition_input(n: int, k: int = 8, seed: int = 0) -> np.ndarray:
    '''A *hard* one-hot community assignment ``C`` (n x k): each node is
    assigned to exactly one of ``k`` communities. The one-hot case is where
    ``relaxed_modularity`` matches the canonical Newman quality score.'''
    rng = np.random.default_rng(seed)
    return np.eye(k, dtype=np.float32)[rng.integers(0, k, n)]


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


def np_coaffiliation(c: Any) -> np.ndarray:
    '''Coaffiliation ``C Cᵀ`` with zeroed diagonal -- the numpy floor + fp64
    oracle (matches nitrix ``coaffiliation`` defaults: exclude_diag, no
    normalise).'''
    c = np.asarray(c)
    out = c @ c.swapaxes(-1, -2)
    return out * (1.0 - np.eye(out.shape[-1], dtype=out.dtype))


def np_relaxed_modularity(a: Any, c: Any, gamma: float = 1.0) -> np.ndarray:
    '''Dense relaxed modularity in nitrix's convention (``exclude_diag=False``,
    undirected): ``Q = (B * C Cᵀ).sum() / 2`` with ``B = (A - gamma kkᵀ/2m) /
    2m``. fp64 oracle. (Equals the canonical Newman quality score / 2 -- see
    module docstring.)'''
    a = np.asarray(a)
    c = np.asarray(c)
    k = a.sum(-1)
    two_m = a.sum()
    b = (a - gamma * np.outer(k, k) / two_m) / two_m
    return ((b * (c @ c.swapaxes(-1, -2))).sum() / 2.0)


def cupy_coaffiliation() -> Callable[[Any], Any]:
    '''GPU coaffiliation ``C Cᵀ`` with zeroed diagonal; cupy lazy.'''

    def run(c: Any) -> Any:
        import cupy as cp

        out = c @ c.swapaxes(-1, -2)
        return out * (1.0 - cp.eye(out.shape[-1], dtype=out.dtype))

    return run


def cupy_relaxed_modularity(gamma: float = 1.0) -> Callable[[Any, Any], Any]:
    '''GPU dense relaxed modularity (nitrix convention); cupy lazy.'''

    def run(a: Any, c: Any) -> Any:
        import cupy as cp

        k = a.sum(-1)
        two_m = a.sum()
        b = (a - gamma * cp.outer(k, k) / two_m) / two_m
        return (b * (c @ c.swapaxes(-1, -2))).sum() / 2.0

    return run


def nx_relaxed_modularity(gamma: float = 1.0) -> Callable[[Any, Any], Any]:
    '''``networkx.community.modularity`` (weighted Newman quality score) on the
    hard partition recovered from ``C`` (argmax), divided by 2 to bridge to
    nitrix's convention (verified exact in fp64; see module docstring).
    networkx lazy (numpy worker only). Builds the Graph from the array (the
    honest end-to-end graph-object cost).'''

    def run(a: Any, c: Any) -> Any:
        import networkx as nx

        a = np.asarray(a)
        labels = np.asarray(c).argmax(-1)
        g = nx.from_numpy_array(a)
        comms = [s for j in range(np.asarray(c).shape[-1])
                 if (s := set(np.where(labels == j)[0]))]
        q = nx.algorithms.community.modularity(
            g, comms, weight='weight', resolution=gamma)
        return q / 2.0

    return run


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
