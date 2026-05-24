# -*- coding: utf-8 -*-
"""Sparse message-passing case (P2): ``semiring_ell_edge_aggregate``.

This is the case PyG was *made* to be compared against.  nitrix's
``semiring_ell_edge_aggregate`` is message passing in the literal sense: for
each vertex it gathers its ELL neighbours, builds a per-edge message via a
user ``edge_fn``, and reduces the messages under a semiring -- exactly the
``message`` / ``aggregate`` split of ``torch_geometric.nn.MessagePassing``.
So unlike the dense ``semiring_matmul`` (where a PyG baseline would be a
strawman), here PyG is the *natural external reference*.

The op under test, per vertex ``i`` with ELL neighbours ``j = indices[i, p]``::

    out[i] = (+)_p  values[i, p] · (W · x[j])          # a GCN-style edge_fn

We use a **linear** ``edge_fn`` (``w · (W @ h_j)``) on purpose: it is the same
math in JAX, in torch, and in the fp64 oracle, so the fidelity gate measures
pure round-off rather than a framework's choice of nonlinearity.  Two algebras
exercise the two reductions the op (and PyG) both support:

- ``real`` -> sum-aggregation (the GCN / GraphSAGE-mean numerator), PyG
  ``aggr='add'``;
- ``tropical_max_plus`` -> max-aggregation (GraphSAGE-max / PointNet), PyG
  ``aggr='max'``.

(``log`` / ``euclidean`` are *not* offered: nitrix's edge-aggregate first cut
only reduces REAL / TROPICAL under a single axis primitive -- the pytree-state
monoids are a deferred follow-up, per ``semiring/ell_edge.py``.)

Baselines: ``nitrix-jax`` (the reference, ratios taken against it) and ``pyg``
(the torch ``MessagePassing`` a practitioner writes).  There is no Pallas path
for edge-aggregate (the kernel inlines an arbitrary ``edge_fn``; not shipped),
and no dense/naive baseline -- the whole point is the sparse adjacency.

Graph: a synthetic regular-degree graph (each row has ``k`` neighbours drawn
uniformly with replacement -- duplicates aggregate identically in both
frameworks, and every vertex is a target, so no empty-row max edge cases).
The same flat ``(indices, values)`` feed nitrix's ELL and PyG's COO
``edge_index`` (``j -> i``, ``flow='source_to_target'``), so both frameworks
see the identical graph + weights + features.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.semiring import (
    REAL,
    TROPICAL_MAX_PLUS,
    semiring_ell_edge_aggregate,
)
from nitrix.sparse import ELL

from ._base import BuiltPoint, Case

# Algebra -> (nitrix semiring, numpy axis-reduce, PyG aggr string).
_ALGEBRAS: Dict[str, Tuple[Any, Callable[[Any], Any], str]] = {
    'real': (REAL, lambda e: e.sum(axis=1), 'add'),
    'tropical_max_plus': (TROPICAL_MAX_PLUS, lambda e: e.max(axis=1), 'max'),
}


def _gcn_edge_fn(W: Any) -> Callable[..., Any]:
    '''Linear GCN-style message: ``e = w · (W @ h_j)`` (ignores h_i / ij).'''

    def edge_fn(h_i: Any, h_j: Any, w: Any, ij: Any) -> Any:
        return w * (W @ h_j)

    return edge_fn


def _pyg_fn(
    idx: np.ndarray, val: np.ndarray, W: np.ndarray, aggr: str
) -> Callable[[Any], Any]:
    '''Build the PyG message-passing baseline (torch imported lazily).

    The static graph (COO ``edge_index``, edge weights, ``W``) is built once
    and cached in the closure on the first call, so the timed steady state is
    just the linear ``x @ Wᵀ`` and the propagate -- not graph construction /
    H2D (that lands in the one-off first-call ``compile_time``).
    '''
    state: Dict[str, Any] = {}

    def run(x: Any) -> Any:
        import torch
        from torch_geometric.nn import MessagePassing

        if not state:
            class _GCNMessage(MessagePassing):
                def __init__(self) -> None:
                    super().__init__(aggr=aggr, flow='source_to_target')

                def forward(self, h: Any, ei: Any, ew: Any) -> Any:
                    return self.propagate(ei, x=h, edge_weight=ew)

                def message(self, x_j: Any, edge_weight: Any) -> Any:
                    return edge_weight.view(-1, 1) * x_j

            n, k = idx.shape
            dev = x.device
            src = torch.as_tensor(idx.reshape(-1), dtype=torch.long,
                                  device=dev)            # neighbour j
            dst = torch.arange(n, device=dev).repeat_interleave(k)  # target i
            state['ei'] = torch.stack([src, dst])        # j -> i
            state['ew'] = torch.as_tensor(val.reshape(-1), dtype=x.dtype,
                                          device=dev)
            state['Wt'] = torch.as_tensor(W, dtype=x.dtype, device=dev)
            state['mp'] = _GCNMessage().to(dev)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
        h = x @ state['Wt'].t()                          # (n, d_out)
        return state['mp'](h, state['ei'], state['ew'])

    return run


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, k = param['n'], param['k']
    d_in, d_out = param['d_in'], param['d_out']
    name = param['algebra']
    alg, np_reduce, aggr = _ALGEBRAS[name]

    rng = np.random.default_rng(param.get('seed', 0))
    x = rng.standard_normal((n, d_in)).astype(np.float32)
    W = (rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    idx = rng.integers(0, n, size=(n, k)).astype(np.int32)
    val = rng.standard_normal((n, k)).astype(np.float32)

    # On-device jax inputs; block so H2D is outside the timed region.
    jx = jax.block_until_ready(jnp.asarray(x))
    jW = jnp.asarray(W)
    identity = 0.0 if alg is REAL else -jnp.inf
    ell = ELL(values=jnp.asarray(val), indices=jnp.asarray(idx),
              n_cols=n, identity=identity)

    # fp64 oracle: the identical linear message + reduce, in pure numpy double.
    h64 = x.astype(np.float64) @ W.astype(np.float64).T   # (n, d_out)
    e64 = val.astype(np.float64)[:, :, None] * h64[idx]   # (n, k, d_out)
    ref = np_reduce(e64)                                   # (n, d_out)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'torch':
            import torch

            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            return (torch.from_numpy(x).to(dev),)
        return (jx,)

    def nitrix_fn(xx: Any) -> Any:
        return semiring_ell_edge_aggregate(
            _gcn_edge_fn(jW), ell, xx, semiring=alg,
        )

    baselines = {
        'nitrix-jax': ('jax', nitrix_fn),
        'pyg': ('torch', _pyg_fn(idx, val, W, aggr)),
    }
    return BuiltPoint(
        baselines=baselines,
        inputs_for=inputs_for,
        fp64_reference=ref,
        ratio_reference='nitrix-jax',
    )


# Graph-sized ladder: d_in = d_out = 64, degree 16, two vertex counts.  The
# message tensor is O(n·k·d_out) -- 64 MB at 16384³ -- so this stays modest.
_SHAPES = [
    {'n': 4096, 'k': 16, 'd_in': 64, 'd_out': 64},
    {'n': 16384, 'k': 16, 'd_in': 64, 'd_out': 64},
]

CASE = Case(
    name='ell_edge_aggregate',
    op_qualname='nitrix.semiring.semiring_ell_edge_aggregate',
    # out[i] depends only on row i's neighbours -> bounded input subset; the
    # fp64 oracle is computed in full (it is cheap), annex §C.
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[
        {**shape, 'algebra': name, 'seed': 0}
        for shape in _SHAPES
        for name in _ALGEBRAS
    ],
    # Decision anchor: the larger graph, sum-aggregation (the GCN common case).
    representative={'n': 16384, 'k': 16, 'd_in': 64, 'd_out': 64,
                    'algebra': 'real', 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
