# -*- coding: utf-8 -*-
"""Scale-gaming defence (the size tier + scaling report).

A perf win at a small benched size is meaningless if a worse asymptotic or
memory growth loses -- or OOMs -- before brain scale.  The defence is the
**scaling curve + the stated cost law**, measured over a brain-scale size tier
(``Case.large_param_points``) kept distinct from the small dev/representative
anchor, and surfaced by ``tools/scaling_report.py`` (speed crossover, HBM
multiplier, projected OOM, OOM-as-signal).  These checks pin that machinery on
the template op (distance_transform / EDT) and its replication to morphology
(the OOM exemplar -- disk/ball im2col OOMs at 256^3).
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.morphology import dilate as nx_dilate
from nitrix.morphology import distance_transform

from nperf import measure
from nperf.cases import distance_transform as dt
from nperf.cases._distance import blob_stack, scipy_edt_batched
from nperf.cases._eigenmap import sparse_graph_ell
from nperf.cases._morphology import _GREY, disk_footprint, morph_stack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import scaling_report as sr  # noqa: E402

_P = 'jax-cuda12'
_CUPY = 'cupyx.scipy.ndimage.distance_transform_edt'


def _row(case, param, baseline, t, hbm, status='ok'):
    return {
        'case': case, 'param_point': param, 'baseline': baseline,
        'platform': _P, 'status': status,
        'metrics': {
            'steady_time': ({'min': t} if t is not None else {}),
            'peak_hbm': {'value': hbm},
        },
    }


def test_edt_declares_size_tier_and_cost_law():
    case = dt.CASE
    # The brain-scale tier exists and carries the derived cost law.
    assert case.large_param_points
    assert case.complexity and 'O(n^' in case.complexity
    # The representative is the *small* dev/drift anchor, not a large point.
    assert case.representative in case.param_points
    assert case.representative not in case.large_param_points
    # The tier reaches well past the representative and includes a cohort batch
    # (the axis where the per-volume HBM cost compounds toward OOM).
    rep_size = sr._size_elems(case.representative)
    big = max(sr._size_elems(p) for p in case.large_param_points)
    assert big > 8 * rep_size
    assert any(p.get('batch') for p in case.large_param_points)


def test_batched_edt_vmap_equals_per_image():
    # The batched (cohort) regime is vmap-per-image; it must equal the
    # per-image reference (the supported batch contract the scaling rows ride).
    stack = blob_stack(3, [40, 40], seed=0)
    vm = np.asarray(jax.vmap(distance_transform)(jnp.asarray(stack)),
                    np.float64)
    per = scipy_edt_batched(stack)
    assert np.abs(vm - per).max() < 1e-3


def test_scaling_finds_losses_and_hbm():
    # Synthetic curve: nitrix wins at 64^3, the baseline overtakes at 256^3
    # with a 5x HBM multiplier -- the analysis must record the win, the
    # large-size loss, and project nitrix to OOM sooner than the baseline.
    case = dt.CASE
    rows = []
    for shape, nt, bt, nh, bh in [
        ([64, 64, 64], 2.0e-4, 3.0e-4, 36.0, 1.0),       # nitrix wins
        ([256, 256, 256], 7.0e-3, 6.0e-3, 335.0, 67.0),  # baseline wins, 5x
    ]:
        p = {'shape': shape, 'seed': 0}
        rows.append(_row(case.name, p, 'nitrix-jax', nt, nh))
        rows.append(_row(case.name, p, _CUPY, bt, bh))
    a = sr._analyse(case, sr._collect(rows, case, _P))
    assert {r['label'] for r in a['wins']} == {'64x64x64'}
    assert {r['label'] for r in a['losses']} == {'256x256x256'}
    assert a['largest']['label'] == '256x256x256'      # sorted by elements
    assert a['largest']['hbm_mult'] == pytest.approx(5.0, rel=0.02)
    assert a['proj_oom_nitrix'] < a['proj_oom_base']  # OOMs at fewer elements


def test_scaling_oom_is_a_signal_not_a_hidden_row():
    # An OOM (nitrix) while a baseline ran is a first-class, elevated outcome.
    case = dt.CASE
    p = {'shape': [512, 512, 512], 'seed': 0}
    rows = [
        _row(case.name, p, 'nitrix-jax', None, None, status='oom'),
        _row(case.name, p, _CUPY, 0.05, 200.0),
    ]
    a = sr._analyse(case, sr._collect(rows, case, _P))
    assert a['ooms'] and a['ooms'][0]['nitrix_status'] == 'oom'


def test_approximate_baseline_excluded_from_crossover():
    # The crossover compares against the best *exact* baseline; a declared
    # ApproxBaseline (Danielsson) must not be picked as "the baseline".
    case = dt.CASE
    p = {'shape': [128, 128, 128], 'seed': 0}
    rows = [
        _row(case.name, p, 'nitrix-jax', 5.0e-4, 58.0),
        _row(case.name, p, _CUPY, 6.0e-4, 8.0),
        _row(case.name, p, 'simpleitk.DanielssonDistanceMap', 1.0e-4, 4.0),
    ]
    a = sr._analyse(case, sr._collect(rows, case, _P))
    assert a['rows'][0]['base_name'] == _CUPY.split('.')[-1]  # cupy, not ITK


# --- morphology replication (the OOM exemplar) -----------------------------

@pytest.mark.parametrize('cname', ['dilate', 'erode', 'open', 'close'])
def test_morphology_declares_size_tier_with_oom_exemplar(cname):
    case = measure.CASES[cname]
    lp = case.large_param_points
    assert lp and all(p.get('tier') == 'large' for p in lp)
    assert case.complexity and 'OOM' in case.complexity
    # a 256^3 ball (the im2col OOM exemplar) and a batched cohort are present.
    assert any(p.get('se') == 'ball' and tuple(p['shape']) == (256, 256, 256)
               for p in lp)
    assert any(p.get('batch') for p in lp)


@pytest.mark.parametrize('cname', ['dilate', 'erode', 'open', 'close'])
def test_morph_large_tier_is_nitrix_plus_cupy_no_oracle(cname):
    # The scale tier measures scale, not fidelity: nitrix + the cupy GPU ref
    # only, and NO fp64 oracle (an O(N*k) CPU oracle at 256^3 is slow + beside
    # the point; correctness is pinned at the dev tier).
    case = measure.CASES[cname]
    p = {'shape': [16, 16, 16], 'se': 'ball', 'radius': 2, 'tier': 'large',
         'seed': 0}
    built = case.build(p)
    assert set(built.baselines) == {'nitrix-jax',
                                    f'cupyx.scipy.ndimage.{_GREY[cname]}'}
    assert built.fp64_reference is None and built.fidelity_note


def test_morph_batched_is_per_image_no_leak():
    # The batched-cohort tier relies on the explicit-SE contract: a rank-d SE
    # on a (B, *spatial) stack windows 1 on the batch axis, so nitrix consumes
    # the stack directly and the result equals a per-image loop (no leak).
    stack = morph_stack(3, [16, 16, 16], seed=0)
    fp = disk_footprint(2, 3)
    se = jnp.asarray(np.where(fp, 0.0, -np.inf).astype(np.float32))
    batched = np.asarray(nx_dilate(jnp.asarray(stack), structuring_element=se))
    per = np.stack([
        np.asarray(nx_dilate(jnp.asarray(stack[i]), structuring_element=se))
        for i in range(3)])
    assert np.abs(batched - per).max() < 1e-4


# --- eigensolver replication (the sparse scale *win*) ----------------------

@pytest.mark.parametrize('cname', ['laplacian_eigenmap',
                                   'diffusion_embedding'])
def test_eigensolver_declares_sparse_size_tier(cname):
    case = measure.CASES[cname]
    lp = case.large_param_points
    assert lp and all(p.get('tier') == 'large' and p.get('fmt') == 'ell'
                      for p in lp)
    # sized by node count n (graph op), reaching fsaverage-scale (n >= ~40k).
    assert all('n' in p and 'shape' not in p for p in lp)
    assert max(p['n'] for p in lp) >= 40000
    assert case.complexity and 'differentiable' in case.complexity


def test_sparse_graph_ell_is_symmetric_and_padded():
    # built directly as ELL (no dense n x n); the underlying adjacency must be
    # symmetric (the Laplacian assumes it) and the ELL width fixed.
    val, idx, A = sparse_graph_ell(500, degree=16, seed=0)
    assert val.shape == idx.shape and val.shape[0] == 500
    diff = (A - A.T)
    assert abs(diff).sum() == 0.0  # symmetric


@pytest.mark.parametrize('cname', ['laplacian_eigenmap',
                                   'diffusion_embedding'])
def test_eigensolver_large_tier_is_nitrix_plus_refs_no_oracle(cname):
    # scale tier: nitrix auto(=lobpcg) + the differentiable vjp + sparse
    # scipy/cupy eigsh refs, and NO fp64 oracle (scale, not fidelity).
    case = measure.CASES[cname]
    p = {'n': 1500, 'degree': 16, 'k': 8, 'fmt': 'ell', 'tier': 'large',
         'seed': 0}
    built = case.build(p)
    assert set(built.baselines) == {
        'nitrix-jax', 'nitrix-jax-lobpcg-vjp',
        'scipy.sparse.eigsh', 'cupyx.sparse.eigsh'}
    assert built.fp64_reference is None and built.fidelity_note
    # nitrix sparse lobpcg runs + the vjp row is finite (the differentiable
    # backward that distinguishes it from eigsh).
    for name in ('nitrix-jax', 'nitrix-jax-lobpcg-vjp'):
        pid, fn = built.baselines[name]
        out = np.asarray(fn(*built.inputs_for('jax')))
        assert out.shape == (8,) and np.isfinite(out).all()


def test_diffusion_sparse_ref_matches_dense_diffusion_operator():
    # The diffusion large tier must score the *diffusion* operator (P_sym,
    # largest-k), not the Laplacian -- the sparse scipy ref on the ELL graph
    # must agree with the dense diffusion oracle on the same adjacency.
    from nperf.cases._diffusion import (
        diffusion_eigenvalues,
        scipy_sparse_eigsh_diffusion,
    )
    val, idx, A = sparse_graph_ell(400, degree=12, seed=1)
    dense = diffusion_eigenvalues(A.toarray(), k=6)
    sparse = scipy_sparse_eigsh_diffusion(k=6)(val, idx, 400)
    assert np.abs(np.asarray(sparse) - dense).max() < 1e-6
