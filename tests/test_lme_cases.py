# -*- coding: utf-8 -*-
"""Tier-2 voxelwise LME case (reml_fit).

CPU build + oracle agreement against the closed-form balanced one-way REML
(the reliable fp64 oracle).  Both baselines run on the host: nitrix (jax, CPU
here) and the looped statsmodels.MixedLM.  Small V so the looped statsmodels
baseline stays quick in the test.
"""
import os
import warnings

import numpy as np
import pytest

from nperf.cases import reml_fit
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of
from nperf.report import economic as ec

_P = {'V': 16, 'k': 8, 'n': 24, 'seed': 0}
# the marquee real-data point + its assembled-arrays cache (skip if absent so
# the suite never triggers the one-time nilearn download).
_REAL_P = {'data': 'localizer', 'V': 512, 'N': 40, 'realism': 'real_full',
           'seed': 0}
_LOC_CACHE = os.path.join(
    os.environ.get('NPERF_REAL_DATA', '/scratch/nperf/real_anatomy'),
    'localizer_reml_40s_512v.npz')


_RSCRIPT = os.environ.get('NPERF_RSCRIPT', '/scratch/nperf/renv/bin/Rscript')


def test_baseline_shape():
    built = reml_fit._build(_P)
    # statsmodels MixedLM + R lme4 lmer (the gold-standard mixed-model REML) +
    # the R.iofloor (CSV/startup floor economic subtracts from R.lme4).
    assert set(built.baselines) == {
        'nitrix-jax', 'statsmodels.MixedLM', 'R.lme4', 'R.iofloor'}
    assert built.ratio_reference == 'nitrix-jax'
    # statsmodels + R are CPU-only references (no GPU LME library).
    assert requires_of(built.baselines['statsmodels.MixedLM'][0]) == 'cpu'
    assert requires_of(built.baselines['R.lme4'][0]) == 'cpu'
    # R.iofloor is a no-op (returns zeros) -> reported, not gated.
    assert any(a.baseline == 'R.iofloor'
               for a in reml_fit.CASE.approximate_baselines)


def test_baselines_match_closed_form_oracle():
    '''The in-process exact baselines (nitrix + statsmodels) match the oracle.
    R baselines are subprocess (validated separately); R.iofloor is a no-op.'''
    built = reml_fit._build(_P)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # statsmodels boundary warnings
        for name, (provider_id, fn) in built.baselines.items():
            if provider_id == 'r':  # subprocess + R.iofloor no-op: see below
                continue
            args = built.inputs_for(framework_of(provider_id))
            fid = compare(fn(*args), built.fp64_reference,
                          rtol=reml_fit.CASE.rtol, atol=reml_fit.CASE.atol)
            assert fid['status'] == 'pass', (
                f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


@pytest.mark.skipif(not os.path.exists(_RSCRIPT), reason='Rscript absent')
def test_r_lme4_matches_oracle():
    '''R lme4 lmer (REML) -- THE gold-standard mixed model -- equals the
    closed-form balanced-one-way oracle (beta + both variance components).'''
    built = reml_fit._build(_P)
    fn = built.baselines['R.lme4'][1]
    out = fn(*built.inputs_for(framework_of('r')))
    fid = compare(out, built.fp64_reference,
                  rtol=reml_fit.CASE.rtol, atol=reml_fit.CASE.atol)
    assert fid['status'] == 'pass', \
        f'R.lme4: rel_to_tol={fid["rel_to_tol"]:.3g}'


def test_statsmodels_is_slow_baseline():
    slow = {s.baseline for s in reml_fit.CASE.slow_baselines}
    assert 'statsmodels.MixedLM' in slow


def test_oracle_columns_are_beta_varb_vare():
    built = reml_fit._build(_P)
    # variance components are non-negative; intercept ~ grand mean (~5).
    ref = built.fp64_reference
    assert (ref[:, 1] >= 0).all() and (ref[:, 2] > 0).all()


def test_real_localizer_build():
    '''The marquee real-data point: real localizer effects grouped by site
    (real_full, no oracle) build with the same baselines + a fidelity note, and
    nitrix runs to a finite beta / non-negative variance components.  Skipped
    if the assembled-arrays cache is absent (no network in CI).'''
    assert ec.realism_rung(_REAL_P) == 'real_full'
    if not os.path.exists(_LOC_CACHE):
        pytest.skip('localizer reml cache absent (pre-warm the helper)')
    built = reml_fit._build(_REAL_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'statsmodels.MixedLM', 'R.lme4', 'R.iofloor'}
    assert built.fp64_reference is None and built.fidelity_note  # no oracle
    out = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    assert out.shape == (_REAL_P['V'], 3)
    assert np.all(np.isfinite(out))
    assert (out[:, 1] >= 0).all() and (out[:, 2] >= 0).all()  # variances >= 0


def test_real_localizer_statsmodels_agreement():
    '''The marquee correctness gate on REAL data: nitrix agrees with
    statsmodels MixedLM -- beta and the within-site variance essentially exact
    (corr > 0.999); the between-site variance correlates strongly but more
    loosely (corr > 0.95) because k=2 sites is a thin, boundary-prone df (the
    documented caveat).  A 256-voxel subset keeps the looped statsmodels quick.
    Skipped without the cache.'''
    if not os.path.exists(_LOC_CACHE):
        pytest.skip('localizer reml cache absent')
    from nperf.cases._lme import statsmodels_reml
    built = reml_fit._build(_REAL_P)
    (full_y,) = built.inputs_for('numpy')
    y = np.asarray(full_y)[:256]
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        built.inputs_for('jax')[0][:256]))
    _, x, _, groups = reml_fit_real_arrays()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sm = statsmodels_reml(y, x, groups)
    assert np.corrcoef(nit[:, 0], sm[:, 0])[0, 1] > 0.999   # beta
    assert np.corrcoef(nit[:, 2], sm[:, 2])[0, 1] > 0.999   # sigma_e^2
    assert np.corrcoef(nit[:, 1], sm[:, 1])[0, 1] > 0.95    # sigma_b^2 (k=2)


def reml_fit_real_arrays():
    from nperf.cases._real_lme import real_reml_localizer
    return real_reml_localizer(_REAL_P['N'], _REAL_P['V'], _REAL_P['seed'])


def test_op_qualname_matches_nitrix():
    assert reml_fit.CASE.op_qualname == 'nitrix.stats.lme.reml_fit'


# -- reml_fit_lowrank: the v2 FaST-LMM low-rank path (N-scaled) --------------
def test_lowrank_contract():
    from nperf.cases import reml_fit_lowrank as lr
    built = lr._build(lr.CASE.representative)
    assert set(built.baselines) == {
        'nitrix-jax', 'statsmodels.MixedLM', 'R.lme4', 'R.iofloor'}
    # same op as reml_fit, distinct case (the low-rank path).
    assert lr.CASE.op_qualname == 'nitrix.stats.lme.reml_fit'
    assert built.fidelity_note and 'fp32' in built.fidelity_note


def test_lowrank_is_exact_in_fp64():
    '''The characterization: low_rank=True is numerically EXACT -- in float64
    it matches the closed-form oracle (the loose gate is only the fp32
    ZZ^T-eigh floor, FR lme-lowrank-eigh-fp32-conditioning).'''
    import jax
    jax.config.update('jax_enable_x64', True)
    try:
        import jax.numpy as jnp
        from nitrix.stats.lme import reml_fit as rf

        from nperf.cases._lme import balanced_oneway, closed_form_reml
        Y, X, Z, _ = balanced_oneway(64, 8, 200, 0)  # N=1600
        ref = closed_form_reml(Y.astype(np.float64), 8, 200)
        r = rf(jnp.asarray(Y, jnp.float64), jnp.asarray(X, jnp.float64),
               jnp.asarray(Z, jnp.float64), low_rank=True)
        out = np.stack([np.asarray(r.beta_hat)[:, 0],
                        np.asarray(r.sigma_b_sq),
                        np.asarray(r.sigma_e_sq)], -1)
        rel = np.max(np.abs(out - ref) / (1e-6 + 1e-6 * np.abs(ref)))
        assert rel <= 1.0, f'low_rank fp64 not exact: rel_to_tol {rel:.3g}'
    finally:
        jax.config.update('jax_enable_x64', False)


def test_lowrank_fp32_passes_loose_gate():
    '''fp32 (the suite policy): low_rank passes the loose REML gate across the
    benched N range (the fp32 floor stays under tolerance up to N=16384).'''
    from nperf.cases import reml_fit_lowrank as lr
    for p in list(lr.CASE.param_points) + list(lr.CASE.large_param_points):
        built = lr._build(p)
        import jax
        out = np.asarray(jax.block_until_ready(
            built.baselines['nitrix-jax'][1](*built.inputs_for('jax'))))
        ref = np.asarray(built.fp64_reference)
        rel = np.max(np.abs(out - ref)
                     / (lr.CASE.atol + lr.CASE.rtol * np.abs(ref)))
        assert rel <= 1.0, f'N={p["N"]}: rel_to_tol {rel:.3g} > 1'
