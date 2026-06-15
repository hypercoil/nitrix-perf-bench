# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.diffusion_embedding`` vs scipy / cupy eigsh.

Coifman-Lafon diffusion-map embedding -- the **largest** nontrivial eigenpairs
of the anisotropic diffusion operator (alpha=0.5).  The sibling of
``laplacian_eigenmap`` and hardened the same way for B18 Win 4:

- ``nitrix-jax`` is the **default** ``solver='auto'`` (``eigh`` for dense,
  which routes to CPU on this L4 -- a cuSOLVER-class issue observed here,
  cause/scope uncharacterised; ``lobpcg`` for sparse) -- the branch users hit,
  not the old lobpcg-pinned headline.  ``lobpcg`` / ``shift_invert`` / ``poly``
  ride as labelled variants.
- ``nitrix-jax-symmetric`` (``promise_symmetry=True``) is the asserted-symmetry
  path: the public DEFAULT ``promise_symmetry=False`` (a bug fix) applies the
  symmetric part ½(A·X + Aᵀ·X) -- two matvecs per lobpcg iteration -- for a
  possibly-non-symmetric *stored* operator; True does one matvec.  The SBM
  input is exactly symmetric, so True is valid + scored against the same fp64
  oracle (a correct baseline, not a cheaper-but-wrong shortcut; it would
  silently diverge on a non-symmetric stored pattern -- pinned by the hazard
  test).  Measured (L4): ~2.5× faster on ELL/lobpcg, ~slower on dense eigh.
- The gate is tight (rtol=atol=1e-4): ``eigh`` / ``lobpcg`` pass;
  ``shift_invert`` (~1e-3) / ``poly`` (~5e-4) are declared ``ApproxBaseline``
  (accuracy reported beside speed -- the tradeoff is the signal).
- Input is a planted-partition (SBM) graph (clustered spectrum); an ELL row
  exercises the matrix-free sparse path (ELL passed as ``(values, indices)``
  and rebuilt inside the jitted baseline -- not a registered pytree, B22).
- ``nitrix-jax-lobpcg-vjp`` times the implicit-VJP backward -- an implicit-
  function-theorem adjoint, *algorithm-independent* (it would wrap an
  implicitly-restarted Lanczos as readily as lobpcg); scipy / cupy ``eigsh``
  ship no adjoint, so there is no twin.

Eigenvectors carry a sign/subspace ambiguity, so the case scores the
**eigenvalues**.  No standard diffusion-map library exists, so the reference is
the operator's own ``scipy.sparse.linalg.eigsh`` (CPU floor) + CuPy ``eigsh``
(GPU ref), scored against an fp64 oracle.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import diffusion_embedding
from nitrix.sparse import ell_from_dense

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._diffusion import (
    _ALPHA,
    cupy_eigsh_diffusion,
    cupy_sparse_eigsh_diffusion,
    diffusion_eigenvalues,
    scipy_eigsh_diffusion,
    scipy_sparse_eigsh_diffusion,
)
from ._eigenmap import build_spectral_large, sbm_input, spectral_baselines


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, k = param['n'], param['k']
    fmt = param.get('fmt', 'dense')

    def op_evals(operand: Any, **kw: Any) -> Any:
        return diffusion_embedding(
            operand, n_components=k, alpha=_ALPHA, t=0.0, **kw)[1]

    if param.get('tier') == 'large':  # brain-graph-scale sparse (no dense)
        return build_spectral_large(
            op_evals, k, param,
            scipy_ref=scipy_sparse_eigsh_diffusion(k),
            cupy_ref=cupy_sparse_eigsh_diffusion(k))

    W = sbm_input(n, seed=param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(W))
    ref = diffusion_eigenvalues(W, k, _ALPHA)  # fp64 oracle (eigenvalues)

    ell_aux: Any = None
    if fmt == 'ell':
        ell = ell_from_dense(jx)
        ell_vals = jax.block_until_ready(ell.values)
        ell_idx = ell.indices
        ell_aux = (ell.n_cols, ell.identity)

        def inputs_for(framework: str) -> Tuple[Any, ...]:
            if framework == 'cupy':
                return to_cupy(W)
            if framework == 'numpy':
                return (W,)
            return (ell_vals, ell_idx)
    else:
        def inputs_for(framework: str) -> Tuple[Any, ...]:
            if framework == 'cupy':
                return to_cupy(W)
            if framework == 'numpy':
                return (W,)
            return (jx,)

    baselines = spectral_baselines(
        op_evals, fmt=fmt, ell_aux=ell_aux,
        scipy_ref=scipy_eigsh_diffusion(k), cupy_ref=cupy_eigsh_diffusion(k),
    )
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# Dense-feasible sizes (the diffusion oracle is dense eigvalsh) + one ELL row.
_POINTS = [
    {'n': 1024, 'k': 8, 'fmt': 'dense'},    # representative
    {'n': 2048, 'k': 8, 'fmt': 'dense'},    # dense scaling
    {'n': 2048, 'k': 8, 'fmt': 'ell'},      # sparse matrix-free path
]

# Brain-graph-scale size tier (scale-gaming defence, COVERAGE_MANDATE §2.6):
# the diffusion sibling of laplacian_eigenmap's sparse tier.  **Sparse** graphs
# at fsaverage5->7 vertex counts, built directly as ELL (no dense n x n -- a
# 100k dense diffusion operator is ~40 GB, unconstructable).  P_sym stays
# sparse under the density rescales, so at these n only the matrix-free path
# exists, and nitrix's is uniquely differentiable.  build_spectral_large with
# the diffusion-operator sparse eigsh refs (largest-k), no oracle (scale).
_LARGE = [
    {'n': 10242, 'degree': 16},    # fsaverage5
    {'n': 40962, 'degree': 16},    # fsaverage6
    {'n': 120000, 'degree': 16},   # toward fsaverage7 (~163k)
]

CASE = Case(
    name='diffusion_embedding',
    op_qualname='nitrix.graph.diffusion_embedding',
    output_independent=False,  # each eigenvalue is a global spectral property
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    large_param_points=tuple(
        {**p, 'k': 8, 'fmt': 'ell', 'tier': 'large', 'seed': 0}
        for p in _LARGE),
    representative={'n': 1024, 'k': 8, 'fmt': 'dense', 'seed': 0},
    # Same scaling law as laplacian_eigenmap: the dense path is O(n^3) eigh /
    # O(n^2) operator -> infeasible at n~100k (~40 GB dense diffusion
    # operator); the sparse lobpcg path is O(iters*nnz) forward + an O(nnz*k)
    # implicit-VJP backward -> scales to fsaverage6/7, and unlike scipy/cupy
    # eigsh it is *differentiable*.  At scale the only question is
    # sparse-vs-sparse + the gradient, not dense-vs-sparse.
    complexity=(
        'dense O(n^3) eigh / O(n^2) operator -> infeasible at n~100k '
        '(~40 GB dense diffusion operator); sparse lobpcg O(iters*nnz) fwd + '
        'O(nnz*k) differentiable backward -> scales (fsaverage6/7), and is '
        'the only differentiable option (scipy/cupy eigsh have no gradient).'
    ),
    build=_build,
    rtol=1e-4,
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'nitrix-jax-shift_invert',
            'inner-CG shift-invert, fixed outer iters -> ~1e-3 eigenvalue '
            'accuracy (converges less far than cupy eigsh); L4 SBM graph.'),
        ApproxBaseline(
            'nitrix-jax-poly',
            'matvec-only polynomial spectral filter -> ~5e-4 eigenvalue '
            'accuracy; the accuracy/speed tradeoff is the signal (L4 SBM).'),
    ),
)
