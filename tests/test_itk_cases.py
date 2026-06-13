# -*- coding: utf-8 -*-
"""Tier-2 SimpleITK-parity cases (histogram_match + n4_bias_field_correction).

These ops target SimpleITK / ITK and have no bit-level fp64 oracle, so the
load-bearing correctness check is **SimpleITK parity** -- re-asserted here with
nitrix's own criteria (it is the no-oracle analogue of the splice-through gate
on `lomb_scargle_interpolate`). Small sizes so the parity tests (which run
SimpleITK's iterative N4) stay quick.
"""
import os

import numpy as np
import pytest

from nperf.cases import dilate, distance_transform, erode, median_filter
from nperf.cases import histogram_match as hm
from nperf.cases import n4_bias_field_correction as n4
from nperf.cases._itk import bias_parity
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of
from nperf.report import economic as ec


def test_histogram_match_baselines():
    built = hm._build({'n': 32, 'seed': 11})
    assert set(built.baselines) == {
        'nitrix-jax', 'simpleitk.HistogramMatching'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is None and built.fidelity_note
    # SimpleITK is a host (numpy-framework) CPU reference, runs everywhere
    assert framework_of(built.baselines['simpleitk.HistogramMatching'][0]) \
        == 'numpy'
    assert requires_of(built.baselines['simpleitk.HistogramMatching'][0]) \
        is None


def test_n4_baselines_and_slow():
    built = n4._build({'s': 32, 'seed': 7})
    assert set(built.baselines) == {'nitrix-jax', 'simpleitk.N4'}
    assert built.fp64_reference is None and built.fidelity_note
    slow = {s.baseline for s in n4.CASE.slow_baselines}
    assert 'simpleitk.N4' in slow


def test_histogram_match_sitk_parity():
    '''The correctness gate: nitrix matches ITK HistogramMatching to within
    1e-3 of the reference intensity range (nitrix's own criterion).'''
    pytest.importorskip('SimpleITK')
    built = hm._build({'n': 32, 'seed': 11})
    src, ref = built.inputs_for('numpy')
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    itk = np.asarray(built.baselines['simpleitk.HistogramMatching'][1](
        src, ref))
    ref_range = float(np.asarray(ref).max() - np.asarray(ref).min())
    assert np.max(np.abs(nit - itk)) < 1e-3 * ref_range


def test_n4_sitk_parity():
    '''The correctness gate: nitrix's corrected image matches ITK N4 globally
    (corr > 0.999, scale-invariant rel-RMSE < 5e-3 over the mask).'''
    pytest.importorskip('SimpleITK')
    built = n4._build({'s': 32, 'seed': 7})
    obs, mask = built.inputs_for('numpy')
    mask = np.asarray(mask)
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    itk = np.asarray(built.baselines['simpleitk.N4'][1](obs, mask))
    corr, rel_rmse = bias_parity(nit, itk, mask)
    assert corr > 0.999, f'corr={corr:.5f}'
    assert rel_rmse < 5e-3, f'rel_rmse={rel_rmse:.4g}'


def test_n4_real_anatomy_build():
    '''The marquee real-data point builds on real MNI152 anatomy with a planted
    bias: same baselines, realism = real_planted, a non-trivial brain mask, and
    nitrix returns a finite, non-negative in-mask correction.  This catches the
    real-input-specific failure modes (empty mask, log-of-nonpositive NaNs)
    cheaply; full ITK parity is the opt-in test below + the authoritative
    sweep.'''
    p = {'data': 'mni152', 'resolution': 2, 'seed': 7}
    assert ec.realism_rung(p) == 'real_planted'
    built = n4._build(p)
    assert set(built.baselines) == {'nitrix-jax', 'simpleitk.N4'}
    obs, mask = built.inputs_for('numpy')
    mask = np.asarray(mask)
    assert 0.05 < float(mask.mean()) < 0.9  # a non-trivial brain mask
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    assert np.all(np.isfinite(nit))
    assert float(nit[mask > 0].min()) >= 0.0  # positive corrected intensities


@pytest.mark.skipif(not os.environ.get('NPERF_REAL_TESTS'),
                    reason='full real-anatomy N4 parity (~23s: ITK N4 on '
                           '~1.1M real voxels); set NPERF_REAL_TESTS=1. Also '
                           'validated each authoritative sweep.')
def test_n4_real_anatomy_sitk_parity():
    '''The marquee correctness gate on REAL data: nitrix's corrected MNI152 T1
    (under a planted bias) matches ITK N4 globally (corr > 0.999, scale-
    invariant rel-RMSE < 5e-3 over the mask) -- the same criterion as the
    synthetic phantom, now on real anatomy.'''
    pytest.importorskip('SimpleITK')
    built = n4._build({'data': 'mni152', 'resolution': 2, 'seed': 7})
    obs, mask = built.inputs_for('numpy')
    mask = np.asarray(mask)
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    itk = np.asarray(built.baselines['simpleitk.N4'][1](obs, mask))
    corr, rel_rmse = bias_parity(nit, itk, mask)
    assert corr > 0.999, f'corr={corr:.5f}'
    assert rel_rmse < 5e-3, f'rel_rmse={rel_rmse:.4g}'


def test_op_qualnames():
    assert hm.CASE.op_qualname == 'nitrix.bias.histogram_match'
    assert n4.CASE.op_qualname == 'nitrix.bias.n4_bias_field_correction'


# ---------------------------------------------------------------------------
# SimpleITK morphology / distance floors added to existing cases
# ---------------------------------------------------------------------------


def test_itk_floor_baselines_registered():
    for case, key in [(erode, 'simpleitk.GrayscaleErode'),
                      (dilate, 'simpleitk.GrayscaleDilate'),
                      (median_filter, 'simpleitk.Median'),
                      (distance_transform,
                       'simpleitk.DanielssonDistanceMap')]:
        built = case._build(case.CASE.representative)
        assert key in built.baselines
        assert framework_of(built.baselines[key][0]) == 'numpy'


def test_itk_floors_match_oracle():
    '''The right-target check: each *exact* ITK floor matches the case's fp64
    oracle (erode/dilate) -- so adding it is a fair comparison, not an
    apples-to-oranges row.  (distance_transform's Danielsson is NOT here: the
    tight EDT gate revealed it is 4SED-approximate, so it is a declared
    ApproxBaseline -- reported not gated -- see test_distance_transform.)
    '''
    pytest.importorskip('SimpleITK')
    cases = [(erode, 'simpleitk.GrayscaleErode'),
             (dilate, 'simpleitk.GrayscaleDilate')]
    for case, key in cases:
        built = case._build(case.CASE.representative)
        out = built.baselines[key][1](*built.inputs_for('numpy'))
        fid = compare(np.asarray(out), built.fp64_reference,
                      rtol=case.CASE.rtol, atol=case.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{key}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_itk_median_floor_matches_interior():
    '''median_filter has no oracle (boundary policies differ); the ITK floor
    matches scipy in the interior -- the perf comparison is task-comparable.'''
    pytest.importorskip('SimpleITK')
    import scipy.ndimage as spnd
    built = median_filter._build({'shape': [64, 64], 'size': 3, 'seed': 0})
    (x,) = built.inputs_for('numpy')
    itk = np.asarray(built.baselines['simpleitk.Median'][1](x))
    sp = spnd.median_filter(np.asarray(x), size=3)
    assert np.max(np.abs((itk - sp)[1:-1, 1:-1])) < 1e-5
