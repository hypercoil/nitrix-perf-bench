# -*- coding: utf-8 -*-
"""Tier-2 voxelwise LME case (reml_fit).

CPU build + oracle agreement against the closed-form balanced one-way REML
(the reliable fp64 oracle).  Both baselines run on the host: nitrix (jax, CPU
here) and the looped statsmodels.MixedLM.  Small V so the looped statsmodels
baseline stays quick in the test.
"""
import warnings

from nperf.cases import reml_fit
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'V': 16, 'k': 8, 'n': 24, 'seed': 0}


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


def test_op_qualname_matches_nitrix():
    assert reml_fit.CASE.op_qualname == 'nitrix.stats.lme.reml_fit'
