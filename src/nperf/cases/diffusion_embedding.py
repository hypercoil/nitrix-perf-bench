# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.diffusion_embedding`` vs scipy / cupy eigsh.

Coifman-Lafon diffusion-map embedding -- the largest nontrivial eigenpairs of
the anisotropic diffusion operator (alpha=0.5). The sibling of
``laplacian_eigenmap``: same two solvers, both benchmarked --
``solver='lobpcg'`` (matrix-free, **genuine GPU**, dodges cuSOLVER) as
``nitrix-jax`` and
``solver='eigh'`` (dense via ``safe_eigh`` -> CPU / wedges on GPU) as
``nitrix-jax-eigh``.

The eigenvectors carry a sign/subspace ambiguity, so the case scores the
**eigenvalues** (returned by the op) against an fp64 numpy oracle. No standard
diffusion-map library exists, so the reference is the operator's own
``scipy.sparse.linalg.eigsh`` (CPU floor) + a CuPy eigsh GPU ref (see
``cases/_diffusion.py``). Ratio vs nitrix-jax (lobpcg).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import diffusion_embedding

from ._base import BuiltPoint, Case, to_cupy
from ._diffusion import (
    _ALPHA,
    cupy_eigsh_diffusion,
    diffusion_eigenvalues,
    scipy_eigsh_diffusion,
)
from ._eigenmap import _K, eigenmap_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    W = eigenmap_input(n, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(W))
    ref = diffusion_eigenvalues(W, _K, _ALPHA)  # fp64 oracle (eigenvalues)

    def evals(a: Any, solver: str) -> Any:
        return diffusion_embedding(
            a, n_components=_K, alpha=_ALPHA, t=0.0, solver=solver)[1]

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W)
        return (W,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda a: evals(a, 'lobpcg')),  # GPU iterative
        'nitrix-jax-eigh': ('jax', lambda a: evals(a, 'eigh')),  # safe_eigh
        'scipy.sparse.eigsh': ('scipy', scipy_eigsh_diffusion()),  # CPU floor
        'cupyx.sparse.eigsh': ('cupy', cupy_eigsh_diffusion()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [512, 1024, 2048]

CASE = Case(
    name='diffusion_embedding',
    op_qualname='nitrix.graph.diffusion_embedding',
    output_independent=False,  # each eigenvalue is a global spectral property
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES],
    representative={'n': 1024, 'seed': 0},
    build=_build,
    # lobpcg is iterative -> looser than the exact solvers (its approximation
    # is the point); the dense / scipy / cupy paths pass tightly.
    rtol=5e-3,
    atol=5e-3,
)
