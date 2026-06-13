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


def test_baseline_shape():
    built = reml_fit._build(_P)
    assert set(built.baselines) == {'nitrix-jax', 'statsmodels.MixedLM'}
    assert built.ratio_reference == 'nitrix-jax'
    # statsmodels is the CPU-only canonical reference (no GPU LME library).
    assert requires_of(built.baselines['statsmodels.MixedLM'][0]) == 'cpu'


def test_baselines_match_closed_form_oracle():
    built = reml_fit._build(_P)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # statsmodels boundary warnings
        for name, (provider_id, fn) in built.baselines.items():
            args = built.inputs_for(framework_of(provider_id))
            fid = compare(fn(*args), built.fp64_reference,
                          rtol=reml_fit.CASE.rtol, atol=reml_fit.CASE.atol)
            assert fid['status'] == 'pass', (
                f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


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
    assert set(built.baselines) == {'nitrix-jax', 'statsmodels.MixedLM'}
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
