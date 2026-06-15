# -*- coding: utf-8 -*-
"""Hardened spectral-embedding cases (B18 Win 4): laplacian_eigenmap /
diffusion_embedding, post the eigensolver rehome to ``linalg._eigsolve``.

Win 4 is the most accuracy-sensitive win: half the iterative speedup *is* lower
accuracy, so it is only real if gated.  These checks pin the seams the perf
rows ride on:

- **default routing**: ``solver='auto'`` resolves to ``eigh`` for dense and
  ``lobpcg`` for sparse -- the branch a user hits, which the headline row
  measures (the old case pinned lobpcg, not the dense default).
- **accuracy is pinned**: ``eigh`` / ``lobpcg`` clear the tight gate;
  ``shift_invert`` / ``poly`` (~1e-3) do not, so they are *declared
  approximate* -- the accuracy/speed tradeoff is reported as a signal, never a
  loose-gate pass (a fast solver that quietly stopped converging fails here).
- **format agreement**: the ELL sparse path computes the same eigenvalues as
  the dense path (so the sparse perf rows are the same op).
- **differentiability**: lobpcg is reverse-mode differentiable (the implicit
  VJP) -- the reason it is the default; scipy / cupy ``eigsh`` have no grad.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.graph import laplacian_eigenmap
from nitrix.sparse import ell_from_dense

from nperf import measure
from nperf.cases import diffusion_embedding as de
from nperf.cases import laplacian_eigenmap as le
from nperf.cases._eigenmap import sbm_input
from nperf.core import Status
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_MODS = [le, de]
# Exact solvers that MUST clear the tight gate against the fp64 oracle at every
# point.  ``nitrix-jax-symmetric`` (promise_symmetry=True) is here because the
# benchmark input is exactly symmetric (sbm_input), so the assumed-symmetry
# fast path computes the SAME correct eigenvalues -- it earns its speed ratio
# against a correct baseline, never by silently computing something cheaper.
# (The precondition + the violated-assumption hazard are pinned separately.)
_EXACT = {'nitrix-jax', 'nitrix-jax-symmetric', 'nitrix-jax-lobpcg',
          'nitrix-jax-lobpcg-vjp', 'scipy.sparse.eigsh'}


def _cpu_attempt(mod, name, param):
    built = mod.CASE.build(param)
    pid, fn = built.baselines[name]
    out = np.asarray(fn(*built.inputs_for(framework_of(pid))), np.float64)
    return out, built, compare(out, built.fp64_reference,
                               rtol=mod.CASE.rtol, atol=mod.CASE.atol)


def test_default_routes_eigh_dense_lobpcg_sparse():
    # The contract the headline row depends on: solver='auto' -> eigh for dense
    # (exact, full spectrum) and lobpcg for sparse.  Asserted on the public op.
    W = sbm_input(512, seed=0)
    jx = jnp.asarray(W)
    auto = np.asarray(laplacian_eigenmap(jx, n_components=8)[1])
    eigh = np.asarray(laplacian_eigenmap(jx, n_components=8, solver='eigh')[1])
    assert np.allclose(auto, eigh, atol=1e-6)            # dense auto IS eigh
    ell = ell_from_dense(jx)
    auto_s = np.asarray(laplacian_eigenmap(ell, n_components=8)[1])
    lob = np.asarray(
        laplacian_eigenmap(ell, n_components=8, solver='lobpcg')[1])
    assert np.allclose(auto_s, lob, atol=1e-5)          # sparse auto IS lobpcg


@pytest.mark.parametrize('mod', _MODS)
def test_exact_solvers_pass_tight_gate(mod):
    # eigh (auto/dense), lobpcg, lobpcg-vjp, scipy must clear the tight gate at
    # every CPU-runnable point -- the accuracy pin the old 5e-3 crutch hid.
    for param in mod.CASE.param_points:
        built = mod.CASE.build(param)
        for name in built.baselines:
            if name not in _EXACT:
                continue
            out, _, fid = _cpu_attempt(mod, name, param)
            assert fid['status'] == 'pass', (
                f'{mod.CASE.name}/{name}@{param}: '
                f'rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


@pytest.mark.parametrize('mod', _MODS)
def test_approx_solvers_are_signal_not_failure(mod):
    # shift_invert / poly are faster by converging less far (~1e-3): they fail
    # the tight gate, and because they are declared ApproxBaseline the row
    # stays OK with an 'approximate' fidelity block and earns a ratio -- the
    # tradeoff is the signal, not a dropped/failed row (Win 4's core guard).
    param = {'n': 512, 'k': 8, 'fmt': 'dense', 'seed': 0}
    case = mod.CASE
    built = case.build(param)
    declared = {a.baseline for a in case.approximate_baselines}
    assert declared == {'nitrix-jax-shift_invert', 'nitrix-jax-poly'}
    recs = [
        measure.measure_attempt(
            case, param, built, b, platform='jax-cpu', run_id='t',
            prov={}, warmup=1, repeats=2)
        for b in ('nitrix-jax', *declared)
    ]
    measure.attach_ratios(recs, built.ratio_reference)
    for r in recs:
        if r.baseline not in declared:
            continue
        assert r.status == Status.OK                    # measured, not failed
        assert r.fidelity['status'] == 'approximate'    # reported, not gated
        assert r.fidelity['rel_to_tol'] > 1.0           # genuinely approximate
        assert r.ratio is not None                      # earns a speed ratio


@pytest.mark.parametrize('mod', _MODS)
def test_dense_and_ell_agree(mod):
    # The ELL sparse path computes the same eigenvalues as the dense path
    # (lobpcg on both), so the sparse perf rows measure the same op.
    p_dense = {'n': 512, 'k': 8, 'fmt': 'dense', 'seed': 0}
    p_ell = {'n': 512, 'k': 8, 'fmt': 'ell', 'seed': 0}
    d, _, _ = _cpu_attempt(mod, 'nitrix-jax-lobpcg-vjp', p_dense)  # dense lob
    bx = mod.CASE.build(p_ell)
    pid, fn = bx.baselines['nitrix-jax']  # ell auto == lobpcg
    e = np.asarray(fn(*bx.inputs_for(framework_of(pid))), np.float64)
    assert np.allclose(d, e, atol=1e-4)


def test_lobpcg_is_differentiable_no_baseline_twin():
    # The differentiability the -vjp row times: grad through lobpcg is finite
    # and nonzero.  scipy/cupy eigsh have no gradient, so the -vjp row is
    # nitrix-only (no baseline twin) -- assert no ref grad twin exists.
    W = jnp.asarray(sbm_input(512, seed=0))

    def loss(A):
        ev = laplacian_eigenmap(A, n_components=8, solver='lobpcg')[1]
        return jnp.sum(ev)

    g = np.asarray(jax.grad(loss)(W))
    assert np.isfinite(g).all() and (g != 0).any()
    built = le.CASE.build({'n': 512, 'k': 8, 'fmt': 'dense', 'seed': 0})
    assert 'nitrix-jax-lobpcg-vjp' in built.baselines
    # no scipy/cupy "vjp" twin exists (the baselines provide no gradient)
    assert not any('vjp' in n and not n.startswith('nitrix')
                   for n in built.baselines)


@pytest.mark.parametrize('mod', _MODS)
def test_gate_stays_tight(mod):
    # Regression guard: the accuracy pin must not loosen back to the historical
    # 5e-3 crutch that let an approximate solver masquerade as exact.
    assert mod.CASE.rtol <= 1e-4
    assert mod.CASE.atol <= 1e-4


def test_cupy_ref_is_gpu_only():
    # The on-target GPU reference is present and platform-gated (needs a GPU).
    for mod in _MODS:
        built = mod.CASE.build({'n': 512, 'k': 8, 'fmt': 'dense', 'seed': 0})
        assert requires_of(built.baselines['cupyx.sparse.eigsh'][0]) == 'gpu'


def test_promise_symmetry_precondition_and_hazard():
    # The nitrix-jax-symmetric variant (promise_symmetry=True) is a speed knob
    # that ASSUMES the operator is symmetric.  Two guards on that assumption:
    #
    # (1) precondition holds: the benchmark adjacency (sbm_input) is EXACTLY
    #     symmetric, so True is valid here -- which is why it sits in _EXACT
    #     and must match the oracle (test_exact_solvers_pass_tight_gate).
    assert float(np.max(np.abs(sbm_input(256, seed=0)
                               - sbm_input(256, seed=0).T))) == 0.0
    #
    # (2) it is NOT a free lunch: on a genuinely non-symmetric *stored* pattern
    #     (top-k kNN -- the op's documented hazard), promise_symmetry=True
    #     silently returns DIFFERENT eigenvalues than the correct symmetrised
    #     (=False) path.  This is the "assumption violated -> silently wrong"
    #     the speed ratio must never be read without.  (NB the False path's own
    #     degree convention is itself under review -- nitrix FR
    #     `laplacian-promise-symmetry-degree`; a dedicated non-symmetric
    #     benchmark case is held pending that decision.)
    rng = np.random.default_rng(1)
    A = rng.random((256, 256)).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    thresh = np.sort(A, axis=1)[:, -8][:, None]
    a_topk = np.where(A >= thresh, A, 0.0).astype(np.float32)  # non-symmetric
    assert float(np.max(np.abs(a_topk - a_topk.T))) > 0.1
    ell = ell_from_dense(jnp.asarray(a_topk))
    correct = np.asarray(laplacian_eigenmap(
        ell, n_components=6, solver='lobpcg', promise_symmetry=False)[1])
    assumed = np.asarray(laplacian_eigenmap(
        ell, n_components=6, solver='lobpcg', promise_symmetry=True)[1])
    assert float(np.max(np.abs(correct - assumed))) > 1e-2  # silently wrong


def test_op_qualnames():
    assert le.CASE.op_qualname == 'nitrix.graph.laplacian_eigenmap'
    assert de.CASE.op_qualname == 'nitrix.graph.diffusion_embedding'
