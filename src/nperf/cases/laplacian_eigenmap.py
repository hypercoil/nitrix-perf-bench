# -*- coding: utf-8 -*-
"""Tier-2 graph: ``nitrix.graph.laplacian_eigenmap`` vs scipy / cupy eigsh.

Spectral (Laplacian-eigenmaps) embedding -- the smallest nontrivial normalised-
Laplacian eigenpairs.  Eigenvectors carry a sign / degenerate-subspace
ambiguity, so the case scores the **eigenvalues** (returned by the op) against
an fp64 oracle.

Hardened for B18 Win 4 (post the eigensolver rehome to ``linalg._eigsolve``):

- **Headline = the default users hit.**  ``nitrix-jax`` is ``solver='auto'`` --
  which the dispatcher resolves to **``eigh`` for dense** (full spectrum via
  ``safe_eigh`` -> CPU on the cuSolver-broken L4) and ``lobpcg`` for sparse.
  The old case pinned ``solver='lobpcg'`` as the headline; that is *not* the
  dense default.  ``lobpcg`` / ``shift_invert`` / ``poly`` ride as labelled
  variants.
- **Accuracy is pinned (the highest-value guard).**  The gate is tight
  (rtol=atol=1e-4); ``eigh`` (exact) and ``lobpcg`` (fp32 runs to the cap,
  ~1e-5/1e-6) pass.  ``shift_invert`` (~1e-3) and ``poly`` (~5e-4) are faster
  *because they converge less far*, so they are declared ``ApproxBaseline`` --
  fidelity reported beside the speed, ratio kept (the accuracy/speed tradeoff
  is the signal, not a loose-gate pass).
- **Realistic spectrum.**  Input is a planted-partition (SBM) graph ->
  clustered low-Laplacian eigenvalues / small spectral gap (connectome-like),
  which stresses iterative convergence; a random/expander graph over-reports.
- **The sparse regime that motivates the op.**  ELL rows at n>dense-feasible
  exercise the matrix-free path (auto->lobpcg; shift_invert / poly are now
  sparse-capable post-refactor).  ELL is passed as ``(values, indices)`` and
  rebuilt inside the jitted baseline -- it is not a registered pytree (nitrix
  B22).
- **The differentiability cost.**  ``nitrix-jax-lobpcg-vjp`` times the
  implicit-VJP backward -- the reason lobpcg is the default; scipy / cupy
  ``eigsh`` provide *no gradient*, so there is no baseline twin for it.

References: ``scipy.sparse.linalg.eigsh`` (CPU floor) + CuPy ``eigsh`` (GPU
ref), both on the same dense ``L_sym`` (so they score the same eigenvalues for
dense and ELL rows alike).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.graph import laplacian_eigenmap
from nitrix.sparse import ell_from_dense

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._eigenmap import (
    build_spectral_large,
    cupy_eigsh,
    laplacian_eigenvalues_oracle,
    sbm_input,
    scipy_eigsh,
    spectral_baselines,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, k = param['n'], param['k']
    fmt = param.get('fmt', 'dense')

    def op_evals(operand: Any, **kw: Any) -> Any:
        return laplacian_eigenmap(operand, n_components=k, **kw)[1]

    if param.get('tier') == 'large':  # brain-graph-scale sparse (no dense)
        return build_spectral_large(op_evals, k, param)

    W = sbm_input(n, seed=param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(W))
    ref = laplacian_eigenvalues_oracle(W, k)  # fp64 oracle (eigenvalues)

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
        scipy_ref=scipy_eigsh(k), cupy_ref=cupy_eigsh(k),
    )
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (n, k, fmt): dense fast-feasible sizes + a k sweep (the win shrinks as k
# grows -> dense eigh favoured) + ELL rows for the sparse regime (n=4096 is
# past dense-eigh comfort; its oracle is scipy eigsh, see _eigenmap).
_POINTS = [
    {'n': 1024, 'k': 8, 'fmt': 'dense'},    # representative
    {'n': 2048, 'k': 8, 'fmt': 'dense'},    # dense scaling
    {'n': 1024, 'k': 32, 'fmt': 'dense'},   # k-dependence
    {'n': 2048, 'k': 8, 'fmt': 'ell'},      # sparse format
    {'n': 4096, 'k': 8, 'fmt': 'ell'},      # large-n sparse (sparse oracle)
]

# Brain-graph-scale size tier (scale-gaming defence, COVERAGE_MANDATE §2.6):
# **sparse** graphs at fsaverage5->7 vertex counts, built directly as ELL (no
# dense n x n -- a 100k dense operator is ~40 GB, unconstructable).  At these n
# only the matrix-free sparse path exists, and nitrix's is uniquely
# differentiable.  build_spectral_large: nitrix lobpcg + the differentiable
# backward + sparse scipy/cupy eigsh refs, no oracle (scale, not fidelity).
_LARGE = [
    {'n': 10242, 'degree': 16},    # fsaverage5
    {'n': 40962, 'degree': 16},    # fsaverage6
    {'n': 120000, 'degree': 16},   # toward fsaverage7 (~163k)
]

CASE = Case(
    name='laplacian_eigenmap',
    op_qualname='nitrix.graph.laplacian_eigenmap',
    output_independent=False,  # each eigenvalue is a global spectral property
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{**p, 'seed': 0} for p in _POINTS],
    large_param_points=tuple(
        {**p, 'k': 8, 'fmt': 'ell', 'tier': 'large', 'seed': 0}
        for p in _LARGE),
    representative={'n': 1024, 'k': 8, 'fmt': 'dense', 'seed': 0},
    # The dense path is O(n^3) (eigh) / O(n^2) (dense lobpcg backward + the n^2
    # operator) -> infeasible at brain-graph scale (a 100k dense operator is
    # ~40 GB).  The sparse lobpcg path is O(iters * nnz) forward + an
    # O(nnz * k) implicit-VJP backward -> scales to fsaverage6/7, and unlike
    # scipy/cupy eigsh it is *differentiable*.  At scale the only question is
    # sparse-vs-sparse + the gradient, not dense-vs-sparse.
    complexity=(
        'dense O(n^3) eigh / O(n^2) backend+operator -> infeasible at '
        'n~100k (~40 GB dense); sparse lobpcg O(iters*nnz) fwd + O(nnz*k) '
        'differentiable backward -> scales (fsaverage6/7), and is the only '
        'differentiable option (scipy/cupy eigsh have no gradient).'
    ),
    build=_build,
    # Tight gate (Win 4: pin accuracy).  eigh / lobpcg / scipy / cupy pass;
    # shift_invert / poly fail it -> declared approximate (signal not failure).
    rtol=1e-4,
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'nitrix-jax-shift_invert',
            'inner-CG shift-invert, fixed outer iters -> ~1e-3 eigenvalue '
            'accuracy (faster by converging less far than cupy eigsh ~1e-10); '
            'measured on the L4 SBM graph.'),
        ApproxBaseline(
            'nitrix-jax-poly',
            'matvec-only polynomial spectral filter, fixed degree/outer iters '
            '-> ~5e-4 eigenvalue accuracy; the accuracy/speed tradeoff is the '
            'signal (L4, SBM graph).'),
    ),
)
