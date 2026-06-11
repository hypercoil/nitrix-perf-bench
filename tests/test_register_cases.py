# -*- coding: utf-8 -*-
"""Registration recipe cases (R1-R2): rigid / affine / diffeomorphic-demons.

Task-level end-to-end drivers: no shared cross-impl oracle (fp64_reference is
None), so these pin (a) the case contract and (b) that each recipe actually
recovers the planted warp -- the accuracy pin the no-oracle bench can't gate.
The recovery tests compile the (unrolled) recipe once each, so they are the
slow tests in this file by design.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.register import (
    DemonsSpec,
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
# affine is xfail(strict): registration-suite-v3 (nitrix 356c768) regressed the
# multi-level GN/LM affine path -- it DIVERGES at this 28^3 size (the coarse
# pyramid level falls to <=14^3; params explode). It recovers fine at >=32^3
# (so the affine *bench* at 96^3+ is unaffected). Filed on nitrix main
# (FR register-affine-small-grid-divergence, 869ca78); strict so the xfail
# flips to a failure -- prompting removal -- once nitrix fixes it.
_RECOVER = [
    pytest.param(rigid_register, RegistrationSpec(levels=2, iterations=15),
                 id='rigid'),
    pytest.param(
        affine_register, RegistrationSpec(levels=2, iterations=15),
        id='affine',
        marks=pytest.mark.xfail(
            reason='v3 affine multi-level GN/LM diverges at 28^3 (coarse '
                   '<=14^3); fine >=32^3 -- nitrix FR '
                   'register-affine-small-grid-divergence (869ca78)',
            strict=True)),
    pytest.param(diffeomorphic_demons_register,
                 DemonsSpec(levels=2, iterations=20), id='demons'),
]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_case_contract(mod):
    built = mod._build(mod.CASE.representative)
    # every recipe carries nitrix + the ANTs/dipy cross-tool refs; the demons
    # case additionally carries the direct ITK demons counterpart.
    common = {'nitrix-jax', 'ants.registration', 'dipy.registration'}
    expected = common | ({'simpleitk.demons'}
                         if mod is demons_mod else set())
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
    assert max(_vox(p) for p in mod.CASE.large_param_points) > _vox(
        mod.CASE.representative)


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
