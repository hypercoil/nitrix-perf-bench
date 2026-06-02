# -*- coding: utf-8 -*-
"""Tier-2 voxelwise FLAME case (flame_two_level).

CPU build + oracle agreement against the constant-within-variance closed-form
FLAME REML.  Only nitrix-jax is a baseline (no fair external competitor); the
closed form is the fidelity oracle.
"""
from nperf.cases import flame_two_level
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_P = {'V': 64, 'N': 60, 'seed': 0}


def test_only_nitrix_baseline():
    built = flame_two_level._build(_P)
    assert set(built.baselines) == {'nitrix-jax'}
    assert built.ratio_reference == 'nitrix-jax'


def test_nitrix_matches_closed_form_oracle():
    built = flame_two_level._build(_P)
    prov, fn = built.baselines['nitrix-jax']
    c = flame_two_level.CASE
    fid = compare(fn(*built.inputs_for(framework_of(prov))),
                  built.fp64_reference, rtol=c.rtol, atol=c.atol)
    assert fid['status'] == 'pass', f'rel_to_tol={fid["rel_to_tol"]:.3g}'


def test_oracle_is_gamma_and_nonneg_varb():
    built = flame_two_level._build(_P)
    ref = built.fp64_reference  # (V, 2) = [gamma, sigma_b^2]
    assert ref.shape == (_P['V'], 2)
    assert (ref[:, 1] >= 0).all()  # variance component non-negative


def test_op_qualname_matches_nitrix():
    assert (flame_two_level.CASE.op_qualname
            == 'nitrix.stats.lme.flame_two_level')
