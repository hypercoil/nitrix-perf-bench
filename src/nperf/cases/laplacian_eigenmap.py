# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.laplacian_eigenmap`` vs scipy / cupy eigsh.

Spectral (Laplacian-eigenmaps) embedding -- the smallest nontrivial normalised-
Laplacian eigenpairs. The eigenvectors carry a sign / degenerate-subspace
ambiguity, so the case scores the **eigenvalues** (returned by the op) against
an fp64 numpy oracle.

The headline is the **solver** axis (see ``cases/_eigenmap.py``): nitrix's two
paths are both benchmarked -- ``solver='lobpcg'`` (matrix-free iterative, runs
**genuinely on the GPU**, dodging the broken cuSOLVER) as ``nitrix-jax``, and
``solver='eigh'`` (dense eigh via ``safe_eigh`` -> **silent CPU fallback** on
this stack) as ``nitrix-jax-eigh``. References: ``scipy.sparse.linalg.eigsh``
(CPU iterative floor) + a CuPy ``eigsh`` GPU ref. Ratio vs nitrix-jax (lobpcg).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import laplacian_eigenmap

from ._base import BuiltPoint, Case, to_cupy
from ._eigenmap import (
    _K,
    cupy_eigsh,
    eigenmap_input,
    laplacian_eigenvalues,
    scipy_eigsh,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    W = eigenmap_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(W))
    ref = laplacian_eigenvalues(W, _K)  # fp64 oracle (eigenvalues)

    def evals(a: Any, solver: str) -> Any:
        return laplacian_eigenmap(a, n_components=_K, solver=solver)[1]

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W)
        return (W,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: evals(a, 'lobpcg')),  # GPU iterative
        'nitrix-jax-eigh': ('jax', lambda a: evals(a, 'eigh')),  # safe_eigh
        'scipy.sparse.eigsh': ('scipy', scipy_eigsh()),  # CPU floor
        'cupyx.sparse.eigsh': ('cupy', cupy_eigsh()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (nodes): the dense-eigh (safe_eigh CPU) path is O(n³); lobpcg's matrix-free
# top-k cost grows far slower, so the GPU-iterative path wins as n grows.
_SIZES = [512, 1024, 2048]

CASE = Case(
    name='laplacian_eigenmap',
    op_qualname='nitrix.graph.laplacian_eigenmap',
    output_independent=False,  # each eigenvalue is a global spectral property
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 1024, 'seed': 0},
    build=_build,
    # lobpcg is iterative -> looser than the exact solvers (its eigenvalue
    # approximation is the point); the dense / scipy / cupy paths pass tightly.
    rtol=5e-3,
    atol=5e-3,
)
