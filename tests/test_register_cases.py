# -*- coding: utf-8 -*-
"""Registration recipe cases (R1-R2): rigid / affine / diffeomorphic-demons.

Task-level end-to-end drivers: no shared cross-impl oracle (fp64_reference is
None), so these pin (a) the case contract and (b) that each recipe actually
recovers the planted warp -- the accuracy pin the no-oracle bench can't gate.
The recovery tests compile the (unrolled) recipe once each, so they are the
slow tests in this file by design.
"""
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.register import (
    MI,
    DemonsSpec,
    MetricForce,
    MIForce,
    RegistrationSpec,
    SyNSpec,
    affine_register,
    diffeomorphic_demons_register,
    greedy_syn_register,
    rigid_register,
)

from nperf.cases import affine_register as affine_mod
from nperf.cases import diffeomorphic_demons as demons_mod
from nperf.cases import greedy_syn_register as syn_mod
from nperf.cases import rigid_register as rigid_mod
from nperf.cases._register import ncc, syn_pair, warp_pair

_MODS = [rigid_mod, affine_mod, demons_mod, syn_mod]
# (recipe fn, spec) per case -- coarse-to-fine, few iters (recovery test).
# affine used to be xfail(strict): registration-suite-v3 (nitrix 356c768)
# regressed the multi-level GN/LM affine path -- it DIVERGED at this 28^3 size
# (coarse pyramid level <=14^3). **RESOLVED in registration-suite-v4** (nitrix
# 58907f8): affine now recovers the planted warp here (the strict xfail flipped
# to XPASS, prompting this removal; FR register-affine-small-grid-divergence).
_RECOVER = [
    pytest.param(rigid_register, RegistrationSpec(levels=2, iterations=15),
                 id='rigid'),
    pytest.param(affine_register, RegistrationSpec(levels=2, iterations=15),
                 id='affine'),
    pytest.param(diffeomorphic_demons_register,
                 DemonsSpec(levels=2, iterations=20), id='demons'),
]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_case_contract(mod):
    built = mod._build(mod.CASE.representative)
    # every recipe carries nitrix + the ANTs/dipy cross-tool refs. demons adds
    # the direct ITK demons counterpart + the exact-SVF 'algebra' variant (the
    # log-Demons parity oracle); SyN adds the two MI-force variants (fMRIPrep's
    # cross-modal metric: closed-form MIForce + autodiff MetricForce(MI)).
    common = {'nitrix-jax', 'ants.registration', 'dipy.registration'}
    extra = set()
    if mod is demons_mod:
        extra = {'simpleitk.demons', 'nitrix-jax-algebra'}
    elif mod is syn_mod:
        extra = {'nitrix-jax-mi', 'nitrix-jax-mi-autodiff'}
    expected = common | extra
    assert set(built.baselines) == expected
    assert built.ratio_reference == 'nitrix-jax'
    # dipy is declared slow on every recipe (skippable via --skip-slow).
    assert 'dipy.registration' in {s.baseline for s in mod.CASE.slow_baselines}
    # task-level: no shared oracle, but a documented reason + the compile law.
    assert built.fp64_reference is None and built.fidelity_note
    assert mod.CASE.complexity and 'compile' in mod.CASE.complexity.lower()
    assert mod.CASE.op_qualname.startswith('nitrix.register.')
    # dev points span (levels, iters) -- post loop-roll, to show compile is
    # flat in iters; the size tier varies the volume (the steady-scaling axis).
    iters = {(p['levels'], p['iters']) for p in mod.CASE.param_points}
    assert len(iters) >= 2
    # brain-scale size tier (COVERAGE_MANDATE §7-D): a larger-volume tier past
    # the representative, for the scaling curve + HBM-headroom projection.
    assert mod.CASE.large_param_points

    def _vox(p):
        return p['shape'][0] * p['shape'][1] * p['shape'][2]
    # the synthetic size tier sweeps the volume past the representative; the
    # real-anatomy points (no 'shape', a fixed real image) are a separate kind.
    shaped = [p for p in mod.CASE.large_param_points if 'shape' in p]
    assert max(_vox(p) for p in shaped) > _vox(mod.CASE.representative)
    # every recipe carries a real-anatomy (MNI152) point alongside synthetic.
    assert any(p.get('data') == 'mni152'
               for p in mod.CASE.large_param_points)


@pytest.mark.parametrize('recipe,spec', _RECOVER)
def test_recipe_recovers_planted_warp(recipe, spec):
    # The accuracy pin: registering moving onto fixed must improve the
    # alignment (each benched recipe does *working* registration, not a
    # degenerate fast-but-wrong one).
    moving, fixed = warp_pair([28, 28, 28], seed=0)
    res = recipe(jnp.asarray(moving), jnp.asarray(fixed), spec=spec)
    before = ncc(moving, fixed)
    after = ncc(np.asarray(res.warped), fixed)
    assert after > before + 0.05, f'no improvement {before:.3f}->{after:.3f}'


def test_syn_recovers_deformation():
    '''greedy SyN on a smooth non-rigid pair: the warp improves alignment AND
    the deformation is diffeomorphic (jacobian_det > 0 everywhere -- no
    folding, the op's stated QA contract).'''
    moving, fixed = syn_pair([28, 28, 28], seed=0)
    res = greedy_syn_register(jnp.asarray(moving), jnp.asarray(fixed),
                              spec=SyNSpec(levels=2, iterations=40))
    before = ncc(moving, fixed)
    after = ncc(np.asarray(res.warped), fixed)
    assert after > before + 0.02, f'no improvement {before:.3f}->{after:.3f}'
    assert bool(jnp.all(res.jacobian_det > 0)), 'folding (jacobian_det <= 0)'


def test_syn_mi_force_recovers_and_parity():
    '''The MI-force variants (fMRIPrep's metric): the closed-form Mattes-MI
    force recovers the deformation (ncc up, diffeomorphic) AND it agrees with
    the autodiff MetricForce(MI()) -- the closed-form-vs-autodiff parity oracle
    the bench's two MI rows measure the *speed* of.'''
    moving, fixed = syn_pair([28, 28, 28], seed=0)
    rm = (float(moving.min()), float(moving.max()))
    rf = (float(fixed.min()), float(fixed.max()))
    spec = SyNSpec(levels=2, iterations=40)
    mv, fx = jnp.asarray(moving), jnp.asarray(fixed)
    r_mi = greedy_syn_register(
        mv, fx, spec=spec,
        force=MIForce(bins=32, range_moving=rm, range_fixed=rf))
    r_ad = greedy_syn_register(
        mv, fx, spec=spec,
        force=MetricForce(MI(bins=32, range_moving=rm, range_fixed=rf)))
    before = ncc(moving, fixed)
    after = ncc(np.asarray(r_mi.warped), fixed)
    assert after > before + 0.03, f'MI weak {before:.3f}->{after:.3f}'
    assert bool(jnp.all(r_mi.jacobian_det > 0)), 'MI folding (jac <= 0)'
    # parity: the closed form is the autodiff direction (nitrix's S3 oracle) --
    # the displacement fields are near-identical (cosine ~1).
    d1 = np.asarray(r_mi.displacement).ravel()
    d2 = np.asarray(r_ad.displacement).ravel()
    cos = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-12))
    assert cos > 0.99, f'closed-form vs autodiff MI diverge (cosine {cos:.4f})'


def test_volreg_contract():
    '''volreg carries nitrix + the community realignment tools (AFNI 3dvolreg /
    FSL mcflirt) + the secondary ANTs moco ref -- no shared oracle; the size
    tier varies T (the batch axis).'''
    from nperf.cases import volreg as volreg_mod
    from nperf.providers import requires_of
    built = volreg_mod._build(volreg_mod.CASE.representative)
    assert set(built.baselines) == {
        'nitrix-jax', 'afni.3dvolreg', 'fsl.mcflirt', 'afni.iofloor',
        'fsl.iofloor', 'ants.motion_correction'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is None and built.fidelity_note
    # every domain ref is a CPU-only binary/tool (the cross-platform bar).
    for name, (prov, _) in built.baselines.items():
        if name != 'nitrix-jax':
            assert requires_of(prov) == 'cpu', name
    # ANTs moco is slow at large T (declared) -- skippable in dev cycles.
    assert 'ants.motion_correction' in {
        s.baseline for s in volreg_mod.CASE.slow_baselines}
    assert volreg_mod.CASE.op_qualname == 'nitrix.register.volreg'
    # the size tier sweeps T (the headline batch axis).
    big_t = {p['T'] for p in volreg_mod.CASE.large_param_points}
    assert max(big_t) > volreg_mod.CASE.representative['T']


def test_volreg_realigns_motion():
    '''The accuracy pin: realigning a motion-corrupted series to its reference
    *reduces* the frame-to-frame variance (the realignment actually aligns).'''
    from nitrix.register import RegistrationSpec, volreg

    from nperf.cases._register import motion_series
    series = motion_series([28, 28, 28], 8, seed=0)
    res = volreg(jnp.asarray(series),
                 spec=RegistrationSpec(levels=2, iterations=15))
    realigned = np.asarray(res.realigned)
    before = float(series.var(axis=0).mean())     # raw inter-frame variance
    after = float(realigned.var(axis=0).mean())   # post-realignment
    assert after < before * 0.9, f'no realignment {before:.4f}->{after:.4f}'


def test_bbr_contract():
    '''bbr is nitrix-only (no ITK/ANTs BBR): a single nitrix baseline, the gap
    documented in the fidelity note; the size tier varies N (the cost axis).'''
    from nperf.cases import bbr_register as bbr_mod
    built = bbr_mod._build(bbr_mod.CASE.representative)
    assert set(built.baselines) == {'nitrix-jax'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is None
    # the no-ITK/ANTs gap is documented (the mandate is unmet by necessity).
    assert 'ITK/ANTs' in built.fidelity_note
    assert bbr_mod.CASE.op_qualname == 'nitrix.register.bbr_register'
    big_n = {p['N'] for p in bbr_mod.CASE.large_param_points}
    assert max(big_n) > bbr_mod.CASE.representative['N']


def test_bbr_recovers_boundary_offset():
    '''The accuracy pin: BBR seats the planted boundary offset back -- the
    final boundary cost is below the initial (the optimiser did work).'''
    from nitrix.register import BBRSpec, bbr_register

    from nperf.cases._register import bbr_boundary
    moving, points, normals = bbr_boundary([32, 32, 32], 1000, seed=0)
    res = bbr_register(jnp.asarray(moving), jnp.asarray(points),
                       jnp.asarray(normals), spec=BBRSpec(iterations=100))
    hist = np.asarray(res.cost_history)
    assert hist[-1] < hist[0], (
        f'cost did not decrease {hist[0]:.4f}->{hist[-1]:.4f}')


@pytest.mark.parametrize('recipe', [rigid_register, affine_register],
                         ids=['rigid', 'affine'])
def test_cross_grid_recovers_via_worldspace(recipe):
    '''Cross-grid (different shape + anisotropic moving spacing) recovered in
    physical space via WorldSpace: the warped moving (on the fixed grid) aligns
    to fixed -- the regime IndexSpace cannot express.'''
    from nitrix.register import WorldSpace

    from nperf.cases._register import warp_pair_cross_grid
    moving, fixed, a_m, a_f = warp_pair_cross_grid(
        [32, 32, 32], [36, 32, 28],
        fixed_spacing=(1, 1, 1), moving_spacing=(1.2, 1.0, 0.9), seed=0)
    res = recipe(jnp.asarray(moving), jnp.asarray(fixed),
                 spec=RegistrationSpec(levels=2, iterations=20),
                 space=WorldSpace(fixed_affine=jnp.asarray(a_f),
                                  moving_affine=jnp.asarray(a_m)))
    after = ncc(np.asarray(res.warped), fixed)
    assert after > 0.6, f'cross-grid registration weak (ncc {after:.3f})'


@pytest.mark.parametrize('recipe,spec', [
    (rigid_register, RegistrationSpec(levels=2, iterations=15)),
    (diffeomorphic_demons_register, DemonsSpec(levels=2, iterations=15)),
], ids=['rigid', 'demons'])
def test_real_anatomy_recovery(recipe, spec):
    '''On REAL anatomy (the MNI152 T1 + a planted warp): the recipe recovers
    (ncc improves) AND the warp is finite. The latter pins the background
    noise floor that breaks the demons-ESM 0/0 NaN on the template's uniform
    background (nitrix FR register-demons-force-divide-by-zero).'''
    pytest.importorskip('nilearn')
    from nperf.cases._real_anatomy import real_syn_pair, real_warp_pair
    moving, fixed = (real_warp_pair(2, 0) if recipe is rigid_register
                     else real_syn_pair(2, 0))
    res = recipe(jnp.asarray(moving), jnp.asarray(fixed), spec=spec)
    warped = np.asarray(res.warped)
    assert np.isfinite(warped).all(), 'NaN warp on real anatomy'
    before, after = ncc(moving, fixed), ncc(warped, fixed)
    assert after > before + 0.02, f'no improvement {before:.3f}->{after:.3f}'


def test_aniso_demons_recovers_warp():
    '''Anisotropic (1x1x3) demons: ``DemonsSpec.spacing`` corrects the bias
    (a voxel-isotropic Gaussian/force is physically anisotropic); the warp
    still improves alignment on the anisotropic grid.'''
    from nperf.cases._register import aniso_pair
    moving, fixed, sp = aniso_pair([28, 28, 28], (1.0, 1.0, 3.0), seed=0)
    res = diffeomorphic_demons_register(
        jnp.asarray(moving), jnp.asarray(fixed),
        spec=DemonsSpec(levels=2, iterations=20, spacing=sp))
    before = ncc(moving, fixed)
    after = ncc(np.asarray(res.warped), fixed)
    assert after > before + 0.02, f'no improvement {before:.3f}->{after:.3f}'


def test_demons_algebra_recovers_matches_group():
    '''The 'algebra' representation (exact-SVF, the parity oracle) recovers the
    planted warp AND lands the same alignment as the default 'group' (perf)
    path -- the two representations are interchangeable in result, so the bench
    measures only their *cost* (group ~2 gathers/iter vs algebra re-exp).'''
    moving, fixed = warp_pair([28, 28, 28], seed=0)
    mv, fx = jnp.asarray(moving), jnp.asarray(fixed)
    spec = DemonsSpec(levels=2, iterations=20)
    r_grp = diffeomorphic_demons_register(mv, fx, spec=spec)
    r_alg = diffeomorphic_demons_register(
        mv, fx, spec=replace(spec, representation='algebra'))
    before = ncc(moving, fixed)
    a_grp = ncc(np.asarray(r_grp.warped), fixed)
    a_alg = ncc(np.asarray(r_alg.warped), fixed)
    assert a_alg > before + 0.05, f'algebra weak {before:.3f}->{a_alg:.3f}'
    assert abs(a_grp - a_alg) < 0.02, (
        f'group/algebra diverge in result ({a_grp:.3f} vs {a_alg:.3f})')


@pytest.mark.parametrize('mod', [syn_mod, demons_mod], ids=['syn', 'demons'])
def test_matched_schedule_builds_and_runs(mod):
    '''The ANTs-canonical size tier uses a per-level iters TUPLE: _build must
    derive levels from its length, enable the early-exit, and wire the matched
    refs (ANTs reg_iterations / dipy level_iters), and the nitrix baseline must
    run finite. A small synthetic tuple point (the real tier is 64^3+).'''
    # the large tier carries the matched tuple-iters points.
    assert any(isinstance(p.get('iters'), (list, tuple))
               for p in mod.CASE.large_param_points)
    p = {'shape': [24, 24, 24], 'levels': 3, 'iters': [8, 4, 2], 'seed': 0}
    built = mod._build(p)
    fn = built.baselines['nitrix-jax'][1]
    out = np.asarray(fn(*built.inputs_for('jax')))
    assert np.isfinite(out).all(), 'matched-schedule nitrix output not finite'
    assert out.shape[-1] == 3, 'expected a (*spatial, ndim) field'
    # the matched path still carries the cross-tool refs (each gets the sched).
    assert {'ants.registration', 'dipy.registration'} <= set(built.baselines)
