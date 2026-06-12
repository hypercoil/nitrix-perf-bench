# -*- coding: utf-8 -*-
"""Tier-2 voxelwise FLAME case (flame_two_level).

CPU build + oracle agreement against the constant-within-variance closed-form
FLAME REML (the fidelity oracle).  Baselines: nitrix-jax, **FSL FLAME**
(``flameo flame1`` -- the fair upstream competitor; skipped if the binary is
absent), and looped ``statsmodels`` meta-analysis (gamma matches; its
Paule-Mandel tau^2 diverges from REML at the boundary by design).
"""
import os

import numpy as np

from nperf.cases import flame_two_level
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'V': 64, 'N': 60, 'seed': 0}
_FLAMEO = os.path.join(
    os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl'), 'bin', 'flameo')


def _run(built, name):
    prov, fn = built.baselines[name]
    return np.asarray(fn(*built.inputs_for(framework_of(prov))))


def test_baseline_set():
    built = flame_two_level._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'fsl.flameo', 'fsl.iofloor',
        'statsmodels.meta_analysis'}
    assert built.ratio_reference == 'nitrix-jax'
    # the external baselines are CPU-only refs (no GPU FLAME library).
    for name in ('fsl.flameo', 'fsl.iofloor', 'statsmodels.meta_analysis'):
        assert requires_of(built.baselines[name][0]) == 'cpu'


def test_external_tools_are_slow_baselines():
    slow = {s.baseline for s in flame_two_level.CASE.slow_baselines}
    assert {'fsl.flameo', 'statsmodels.meta_analysis'} <= slow


def test_nitrix_matches_closed_form_oracle():
    built = flame_two_level._build(_P)
    c = flame_two_level.CASE
    fid = compare(_run(built, 'nitrix-jax'), built.fp64_reference,
                  rtol=c.rtol, atol=c.atol)
    assert fid['status'] == 'pass', f'rel_to_tol={fid["rel_to_tol"]:.3g}'


def test_flameo_matches_oracle():
    '''FSL FLAME is the fair competitor: gamma matches tightly, sigma_b^2 to
    the EM-vs-closed-form floor.  Skipped where FSL is not installed.'''
    if not os.path.exists(_FLAMEO):
        import pytest
        pytest.skip('FSL flameo not installed (see setup_neuro_refs.sh)')
    built = flame_two_level._build(_P)
    out, ref = _run(built, 'fsl.flameo'), built.fp64_reference
    assert np.abs(out[:, 0] - ref[:, 0]).max() < 1e-2   # gamma
    assert np.abs(out[:, 1] - ref[:, 1]).max() < 5e-2   # sigma_b^2


def test_statsmodels_gamma_matches_oracle():
    '''The pooled effect (GLS mean) matches; tau^2 (Paule-Mandel) is NOT REML
    and may hit the tau^2=0 boundary -- a documented divergence, so only gamma
    is asserted here.'''
    built = flame_two_level._build(_P)
    out = _run(built, 'statsmodels.meta_analysis')
    assert np.abs(out[:, 0] - built.fp64_reference[:, 0]).max() < 1e-3


def test_iofloor_is_timing_only():
    '''The fsl.iofloor no-op returns a non-scored placeholder (its value is the
    NIfTI round-trip wall-clock, not an estimate).'''
    built = flame_two_level._build(_P)
    if not os.path.exists(_FLAMEO):
        import pytest
        pytest.skip('FSL fslmaths not installed (see setup_neuro_refs.sh)')
    out = _run(built, 'fsl.iofloor')
    assert out.shape == (_P['V'], 2)


def test_oracle_is_gamma_and_nonneg_varb():
    built = flame_two_level._build(_P)
    ref = built.fp64_reference  # (V, 2) = [gamma, sigma_b^2]
    assert ref.shape == (_P['V'], 2)
    assert (ref[:, 1] >= 0).all()  # variance component non-negative


def test_op_qualname_matches_nitrix():
    assert (flame_two_level.CASE.op_qualname
            == 'nitrix.stats.lme.flame_two_level')
