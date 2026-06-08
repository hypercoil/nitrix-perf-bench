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


def sbm_input(n: int, seed: int = 0, n_blocks: int = 4,
              p_in: float = 0.30, p_out: float = 0.01) -> np.ndarray:
    '''A planted-partition (SBM) weighted symmetric adjacency: community
    structure -> tightly **clustered** low-Laplacian eigenvalues with a small
    spectral gap -- the connectome-like regime that stresses iterative-solver
    convergence and the implicit-VJP ``eps_clamp`` (B18 Win 4: bench a
    realistic spectrum, not a well-separated random/expander graph, which
    over-reports iterative convergence).  A connectivity ring keeps it one
    component (the trivial eigenvector unique).'''
    rng = np.random.default_rng(seed)
    lab = np.zeros(n, dtype=np.int64)
    for b, bl in enumerate(np.array_split(np.arange(n), n_blocks)):
        lab[bl] = b
    P = np.where(lab[:, None] == lab[None, :], p_in, p_out)
    A = (rng.uniform(size=(n, n)) < P).astype(np.float32)
    A = np.triu(A, 1)
    A = A + A.T
    i = np.arange(n)
    nxt = (i + 1) % n
    A[i, nxt] = np.maximum(A[i, nxt], 1.0)  # connectivity ring
    A[nxt, i] = A[i, nxt]
    np.fill_diagonal(A, 0.0)
    return A.astype(np.float32)


def laplacian_eigenvalues_oracle(W: Any, k: int = _K) -> np.ndarray:
    '''fp64 oracle for the k smallest **nontrivial** L_sym eigenvalues.

    Dense ``eigvalsh`` where feasible (``n <= 2048``); for the large-n sparse
    rows (``n > 2048``, where a dense O(n^3) solve is the regime sparse exists
    to avoid) the oracle is scipy ``eigsh`` in fp64 (ARPACK -> ~1e-12, oracle-
    grade for the extremal few) on the same dense-built ``L_sym``, so the gate
    stays tight without materialising a full spectrum.'''
    W64 = np.asarray(W, np.float64)
    n = W64.shape[-1]
    if n <= 2048:
        return laplacian_eigenvalues(W64, k)
    import scipy.sparse.linalg as sla

    lsym = _lsym(W64)
    ev = sla.eigsh(lsym, k=k + 1, which='SA', return_eigenvectors=False)
    return np.sort(ev)[1:k + 1]


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


def spectral_baselines(
    op_evals: Callable[..., Any],
    *,
    fmt: str,
    ell_aux: Any,
    scipy_ref: Callable[[Any], Any],
    cupy_ref: Callable[[Any], Any],
) -> dict:
    '''Assemble the nitrix-jax + scipy + cupy baselines for a spectral case
    (shared by laplacian_eigenmap / diffusion_embedding).

    ``op_evals(operand, **solver_kw)`` returns the eigenvalues (the op's
    ``[1]`` output).  Closures are format-specific so they match the case's
    ``inputs_for``: a **dense** point passes ``(a,)``; an **ELL** point passes
    ``(values, indices)`` and the closure rebuilds ``ELL`` inside the jitted
    fn, because ELL is not a registered JAX pytree and cannot cross the jit
    boundary as an operand (nitrix B22 -- the workaround this carries).

    The headline ``nitrix-jax`` is the *default* (``solver='auto'``): the
    branch a user actually hits -- ``eigh`` for dense (full spectrum via
    ``safe_eigh``, CPU on the cuSolver-broken L4), ``lobpcg`` for sparse.  On
    dense, ``auto != lobpcg`` so a distinct ``lobpcg`` row is added; on ELL
    ``auto`` *is* lobpcg, so no separate row.  ``shift_invert`` / ``poly`` are
    the preconditioned (~1e-3) approximate paths; ``-vjp`` times the implicit-
    VJP backward.  That adjoint is the implicit function theorem on the
    eigenpair (it needs only the converged pairs + the matvec), so it is
    *algorithm-independent* -- it would wrap an implicitly-restarted Lanczos as
    readily as lobpcg; scipy/cupy ``eigsh`` ship no adjoint at all, so there is
    no twin for it.
    '''
    import jax
    import jax.numpy as jnp
    from nitrix.sparse import ELL

    def runner(kw: dict) -> Callable[..., Any]:
        if fmt == 'ell':
            n_cols, identity = ell_aux

            def run_ell(values: Any, indices: Any) -> Any:
                return op_evals(ELL(values, indices, n_cols, identity), **kw)

            return run_ell

        def run_dense(a: Any) -> Any:
            return op_evals(a, **kw)

        return run_dense

    def vjp_runner() -> Callable[..., Any]:
        # value + backward through lobpcg; returns the eigenvalues with the
        # gradient tied in (so XLA cannot DCE the backward) -- values
        # unchanged, timing includes the implicit VJP.
        if fmt == 'ell':
            n_cols, identity = ell_aux

            def run_ell(values: Any, indices: Any) -> Any:
                def f(v: Any) -> Any:
                    return op_evals(
                        ELL(v, indices, n_cols, identity), solver='lobpcg')

                ev, vjp = jax.vjp(f, values)
                (g,) = vjp(jnp.ones_like(ev))
                return ev + (0.0 * jnp.sum(g)).astype(ev.dtype)

            return run_ell

        def run_dense(a: Any) -> Any:
            def f(z: Any) -> Any:
                return op_evals(z, solver='lobpcg')

            ev, vjp = jax.vjp(f, a)
            (g,) = vjp(jnp.ones_like(ev))
            return ev + (0.0 * jnp.sum(g)).astype(ev.dtype)

        return run_dense

    baselines = {
        'nitrix-jax': ('jax', runner({})),  # default: eigh(dense)/lobpcg(ell)
        'nitrix-jax-shift_invert': (
            'jax', runner({'solver': 'shift_invert'})),  # approx ~1e-3
        'nitrix-jax-poly': (
            'jax', runner({'preconditioner': 'polynomial'})),  # approx ~1e-3
        'nitrix-jax-lobpcg-vjp': ('jax', vjp_runner()),  # +backward (no twin)
        'scipy.sparse.eigsh': ('scipy', scipy_ref),  # CPU floor
        'cupyx.sparse.eigsh': ('cupy', cupy_ref),  # GPU ref (no gradient)
    }
    if fmt == 'dense':
        # dense auto -> eigh, so lobpcg is a distinct GPU-iterative variant.
        baselines['nitrix-jax-lobpcg'] = ('jax', runner({'solver': 'lobpcg'}))
    return baselines


# ---------------------------------------------------------------------------
# Scale tier (B23 / scale-gaming): brain-graph-scale **sparse** points.  At
# fsaverage6 (~40k) / fsaverage7 (~160k) a dense n x n operator is
# unconstructable (40 GB+), so only the matrix-free sparse (ELL) path exists --
# and nitrix's is uniquely *differentiable* (the implicit VJP).  The graph is a
# random symmetric expander built directly as ELL (no dense): it has a spectral
# gap, so lobpcg converges and the tier isolates *scaling* (the dev-tier SBM
# covers the clustered-spectrum convergence stress).
# ---------------------------------------------------------------------------


def sparse_graph_ell(n: int, degree: int = 16, seed: int = 0):
    '''A large random symmetric sparse graph as ELL arrays
    ``(values, indices, csr)`` -- built via scipy.sparse, never a dense n x n.
    Each node draws ``degree`` random targets; symmetrised + a ring for
    connectivity; padded to a fixed ELL width (absent slots -> self-index,
    value 0).  ``csr`` is returned for the sparse references.'''
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n), degree)
    cols = rng.integers(0, n, size=n * degree)
    A = sp.csr_matrix(
        (np.ones(n * degree, np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetric
    i = np.arange(n)
    ring = sp.csr_matrix(
        (np.ones(n, np.float32), (i, (i + 1) % n)), shape=(n, n))
    A = A.maximum(ring.maximum(ring.T))
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocsr()
    counts = np.diff(A.indptr)
    k_max = int(counts.max())
    idx = np.tile(i[:, None], (1, k_max)).astype(np.int32)
    val = np.zeros((n, k_max), np.float32)
    col_in_row = (np.arange(A.indices.size)
                  - np.repeat(A.indptr[:-1], counts))
    rr = np.repeat(np.arange(n), counts)
    idx[rr, col_in_row] = A.indices
    val[rr, col_in_row] = A.data
    return val, idx, A


def _lsym_from_csr(A: Any, xp: Any, slinalg: Any) -> Any:
    '''Symmetric normalised Laplacian ``I - D^-1/2 A D^-1/2`` of a sparse CSR,
    in the given array module (scipy or cupyx).'''
    n = A.shape[0]
    d = xp.asarray(A.sum(1)).ravel()
    dm = slinalg.diags(1.0 / xp.sqrt(d))
    return slinalg.eye(n) - dm @ A @ dm


def _csr_from_ell(val: Any, idx: Any, n: int, xp: Any, sp: Any) -> Any:
    '''Rebuild a CSR adjacency from the ELL arrays the bench passes
    (``(values, indices)``); padded slots are self-index / value-0 and sum to a
    harmless diagonal zero.  ``xp`` / ``sp`` are the array + sparse modules
    (numpy + scipy.sparse, or cupy + cupyx.scipy.sparse), so the one builder
    serves both the CPU and GPU sparse references (Laplacian + diffusion).'''
    rows = xp.repeat(xp.arange(n), idx.shape[1])
    return sp.csr_matrix(
        (xp.asarray(val).ravel(), (rows, xp.asarray(idx).ravel())),
        shape=(n, n))


def scipy_sparse_eigsh(k: int = _K) -> Callable[..., Any]:
    '''Smallest-k nontrivial L_sym eigenvalues via scipy **sparse** eigsh on
    the sparse adjacency (the CPU floor that actually scales to n~100k -- a
    dense eigsh would OOM on the operator).'''

    def run(val: Any, idx: Any, n: int) -> Any:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        A = _csr_from_ell(val, idx, n, np, sp)
        lsym = _lsym_from_csr(A, np, sp).tocsc()
        ev = spla.eigsh(lsym, k=k + 1, which='SA', return_eigenvectors=False)
        return np.sort(ev)[1:k + 1]

    return run


def cupy_sparse_eigsh(k: int = _K) -> Callable[..., Any]:
    '''GPU twin via cupyx sparse eigsh; cupy lazy.  Some CuPy builds lack a
    sparse ``eigsh`` -- then this raises and the row records the gap (the
    eigensolver scale tier is honestly GPU-ref-less there, like the chamfer
    case), nitrix is still measured.'''

    def run(val: Any, idx: Any, n: int) -> Any:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as csla

        A = _csr_from_ell(val, idx, n, cp, csp)
        lsym = _lsym_from_csr(A, cp, csp)
        ev = csla.eigsh(lsym, k=k + 1, which='SA', return_eigenvectors=False)
        return cp.sort(ev)[1:k + 1]

    return run


def build_spectral_large(op_evals: Callable[..., Any], k: int, param: dict,
                         *, scipy_ref: Callable[..., Any] = None,
                         cupy_ref: Callable[..., Any] = None) -> Any:
    '''A ``BuiltPoint`` for a brain-graph-scale **sparse** point.  Baselines:
    nitrix ``auto`` (= lobpcg on sparse), ``-vjp`` (the implicit-adjoint
    backward -- algorithm-independent, so it differentiates any forward solver;
    scipy/cupy eigsh ship none), and the sparse scipy (CPU) / cupy (GPU)
    ``eigsh`` refs.  No fp64 oracle
    (``None`` -> inconclusive): this tier measures *scale* -- whether the
    sparse path runs at fsaverage6/7 sizes where a dense operator OOMs;
    lobpcg's iterative accuracy is characterised at the dev tier.  ELL is
    passed as ``(values, indices)`` and rebuilt inside the jitted baseline
    (not a registered pytree, B22).

    ``scipy_ref`` / ``cupy_ref`` are the operator's sparse eigsh references --
    they take ``(values, indices, n)`` and rebuild the sparse operator on that
    adjacency.  They default to the Laplacian (``L_sym``, smallest-k) refs;
    diffusion_embedding passes the diffusion-operator (``P_sym``, largest-k)
    refs so each case scores the *same* operator nitrix computes.'''
    import jax
    import jax.numpy as jnp
    from nitrix.sparse import ELL

    from ._base import BuiltPoint, to_cupy

    if scipy_ref is None:
        scipy_ref = scipy_sparse_eigsh(k)
    if cupy_ref is None:
        cupy_ref = cupy_sparse_eigsh(k)

    n = int(param['n'])
    val, idx, _ = sparse_graph_ell(n, int(param.get('degree', 16)),
                                   param.get('seed', 0))
    jv = jax.block_until_ready(jnp.asarray(val))
    ji = jax.block_until_ready(jnp.asarray(idx))

    def nx(**kw: Any) -> Callable[..., Any]:
        def run(v: Any, i: Any) -> Any:
            return op_evals(ELL(v, i, n, 0.0), **kw)
        return run

    def nx_vjp() -> Callable[..., Any]:
        def run(v: Any, i: Any) -> Any:
            def f(vv: Any) -> Any:
                return op_evals(ELL(vv, i, n, 0.0), solver='lobpcg')
            ev, vjp = jax.vjp(f, v)
            (g,) = vjp(jnp.ones_like(ev))
            return ev + (0.0 * jnp.sum(g)).astype(ev.dtype)
        return run

    def inputs_for(framework: str):
        if framework == 'cupy':
            return to_cupy(val, idx) + (n,)
        if framework == 'numpy':
            return (val, idx, n)
        return (jv, ji)

    baselines = {
        'nitrix-jax': ('jax', nx()),                       # auto -> lobpcg
        'nitrix-jax-lobpcg-vjp': ('jax', nx_vjp()),       # +backward (no twin)
        'scipy.sparse.eigsh': ('scipy', scipy_ref),        # CPU floor
        'cupyx.sparse.eigsh': ('cupy', cupy_ref),          # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('scale tier: sparse-path scaling at brain-graph sizes; '
                       'lobpcg accuracy characterised at the dev tier'),
        ratio_reference='nitrix-jax',
    )
