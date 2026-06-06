# -*- coding: utf-8 -*-
"""Hardened distance_transform cases (B18 Win 1): euclidean + chamfer.

The euclidean default is now an *exact* EDT, so it is gated *tight* (vs
scipy/ITK EDT) -- this file guards that the old ``atol=1.0`` crutch cannot
creep back (it would hide an exact->approximate regression). The chamfer
branch is exact for its own metric (chessboard), gated tight vs scipy ``cdt``.
Two structural anti-gaming checks accompany them:

- **anisotropic gap**: nitrix bakes unit spacing (no ``sampling=``), so it
  matches scipy EDT *without* sampling but diverges from anisotropic scipy --
  a documented feature gap, asserted so it is visible.
- **batch contract**: nitrix treats every axis as spatial (scipy convention),
  so ``distance_transform`` of a stack is a 3-D transform (distances leak
  across the batch); the supported batching path is ``vmap``, asserted to equal
  a per-image loop.
"""
import jax
import numpy as np
import pytest
import scipy.ndimage as spnd
from nitrix.morphology import distance_transform as nx_dt

from nperf import measure
from nperf.cases import distance_transform, distance_transform_chamfer
from nperf.cases._distance import blob_mask
from nperf.core import Status
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_CASES = [
    (distance_transform, {'shape': [48, 48], 'seed': 0}),
    (distance_transform, {'shape': [24, 24, 24], 'seed': 0}),
    (distance_transform_chamfer, {'shape': [48, 48], 'seed': 0}),
    (distance_transform_chamfer, {'shape': [24, 24, 24], 'seed': 0}),
]


@pytest.mark.parametrize('mod,param', _CASES)
def test_exact_baselines_match_oracle_tight(mod, param):
    # The exact references (nitrix + scipy/cupy EDT/cdt) must pass the tight
    # gate; declared-approximate baselines (Danielsson) are exempt -- their
    # gap is the signal, checked separately below.
    built = mod._build(param)
    assert built.fp64_reference is not None
    approx = {a.baseline for a in mod.CASE.approximate_baselines}
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu' or name in approx:
            continue  # gpu ref needs a device; approximate refs aren't gated
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))),
                         dtype=np.float64)
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_approximate_baseline_is_signal_not_failure():
    # The harness contract for ApproxBaseline: an approximate reference
    # (Danielsson 4SED) is measured, its fidelity is reported as 'approximate'
    # (not 'fidelity_failed'), and it still earns a ratio -- so the
    # accuracy/speed tradeoff is a visible signal, not a dropped row.
    case = measure.CASES['distance_transform']
    param = {'shape': [48, 48, 48], 'seed': 0}
    built = case.build(param)
    name = 'simpleitk.DanielssonDistanceMap'
    assert name in {a.baseline for a in case.approximate_baselines}
    recs = [
        measure.measure_attempt(
            case, param, built, b, platform='jax-cpu', run_id='t',
            prov={}, warmup=1, repeats=2)
        for b in ('nitrix-jax', name)
    ]
    measure.attach_ratios(recs, built.ratio_reference)
    dan = next(r for r in recs if r.baseline == name)
    assert dan.status == Status.OK                       # measured, not failed
    assert dan.fidelity['status'] == 'approximate'      # reported, not gated
    assert dan.fidelity['rel_to_tol'] > 1.0             # genuinely approximate
    assert dan.ratio is not None                        # earns a speed ratio


def test_euclidean_gate_stays_tight():
    # Regression guard: the exact EDT must keep a tight gate; the historical
    # atol=1.0 (quasi-Euclidean crutch) must not creep back and re-open the
    # exact->approximate hole.
    assert distance_transform.CASE.atol <= 1e-2
    assert distance_transform.CASE.rtol <= 1e-2


def test_chamfer_has_no_cupy_ref_recorded():
    # cupyx.scipy.ndimage has no distance_transform_cdt; the chamfer case is
    # honestly GPU-ref-less (vs the euclidean case which carries the cupy EDT).
    built = distance_transform_chamfer._build({'shape': [32, 32], 'seed': 0})
    assert not any(requires_of(p) == 'gpu'
                   for p, _ in built.baselines.values())
    eucl = distance_transform._build({'shape': [32, 32], 'seed': 0})
    assert any(requires_of(p) == 'gpu' for p, _ in eucl.baselines.values())


def test_anisotropic_gap_is_visible():
    # nitrix euclidean bakes unit spacing: matches scipy EDT without sampling,
    # diverges from anisotropic scipy (the missing-`sampling=` feature gap).
    m = blob_mask((32, 32, 32), seed=1)
    nx = np.asarray(nx_dt(m), np.float64)
    ref_unit = spnd.distance_transform_edt(m > 0.5)
    ref_aniso = spnd.distance_transform_edt(m > 0.5, sampling=(1, 1, 3))
    assert np.abs(nx - ref_unit).max() < 1e-3      # unit-spacing contract
    assert np.abs(nx - ref_aniso).max() > 0.5      # the gap, made explicit


def test_batch_contract_vmap_equals_per_image():
    # All-axes-spatial: a stack is transformed as 3-D (distances leak across
    # the batch); vmap is the supported per-image batching path.
    stack = np.stack([blob_mask((40, 40), seed=s) for s in range(4)])
    vm = np.asarray(jax.vmap(nx_dt)(stack), np.float64)
    per = np.stack([spnd.distance_transform_edt(stack[i] > 0.5)
                    for i in range(len(stack))])
    all_spatial = np.asarray(nx_dt(stack), np.float64)  # 3-D, leaks
    assert np.abs(vm - per).max() < 1e-3          # vmap == per-image
    assert np.abs(all_spatial - per).max() > 0.5  # the leak, made explicit


def test_op_qualnames():
    assert (distance_transform.CASE.op_qualname
            == 'nitrix.morphology.distance_transform')
    assert (distance_transform_chamfer.CASE.op_qualname
            == 'nitrix.morphology.distance_transform')
