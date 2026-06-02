# -*- coding: utf-8 -*-
"""Tier-2 precision / partial-covariance cases (precision / partialcov /
partialcorr).

CPU build + oracle agreement for the runnable (jax + numpy) baselines; the cupy
GPU ref is skipped here (needs a device + the refs env).  These are inverse-
covariance ops; nitrix's jitted inv lowers off cuSolver and runs on GPU, but
the cupy ref's cupy.linalg.inv fails at large c -- so the cupy ref is GPU-only
(skipped here for lack of a device, with the eigh-family GPU caveat).
"""
import pytest

from nperf.cases import partialcorr, partialcov, precision
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# obs >> c so nilearn's 1/n (MLE) vs nitrix's 1/(n-1) covariance ddof rides
# under tolerance for precision (the difference is ~ n/(n-1); see _precision).
_P = {'c': 32, 'obs': 2048, 'seed': 0}
_CASES = [
    (precision, 'numpy.inv_cov'),
    (partialcov, 'numpy.partialcov'),
    (partialcorr, 'numpy.partialcorr'),
]


@pytest.mark.parametrize('mod,refname', _CASES)
def test_baseline_shape(mod, refname):
    built = mod._build(_P)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    # extras are a GPU-only on-target ref (cupy) or a CPU floor (the nilearn
    # community-standard estimator, added to precision + partialcorr).
    for extra in names - {'nitrix-jax', refname}:
        prov = built.baselines[extra][0]
        assert requires_of(prov) == 'gpu' or framework_of(prov) == 'numpy'


@pytest.mark.parametrize('mod,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, refname):
    built = mod._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        args = built.inputs_for(framework_of(provider_id))
        fid = compare(fn(*args), built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_nilearn_floor_registered():
    '''precision + partialcorr carry the nilearn community-standard floor
    (EmpiricalCovariance); partialcov has no nilearn partial-cov kind.'''
    assert 'nilearn.precision' in set(precision._build(_P).baselines)
    pc = set(partialcorr._build(_P).baselines)
    assert 'nilearn.partial_correlation' in pc
    prov = precision._build(_P).baselines['nilearn.precision'][0]
    assert framework_of(prov) == 'numpy'
    assert not any('nilearn' in b for b in partialcov._build(_P).baselines)


def test_partialcorr_unit_diagonal():
    '''partialcorr has a unit diagonal (it is a correlation matrix).'''
    import numpy as np
    built = partialcorr._build(_P)
    np.testing.assert_allclose(np.diag(built.fp64_reference), 1.0, atol=1e-9)


def test_op_qualnames_match_nitrix():
    assert precision.CASE.op_qualname == 'nitrix.stats.precision'
    assert partialcov.CASE.op_qualname == 'nitrix.stats.partialcov'
    assert partialcorr.CASE.op_qualname == 'nitrix.stats.partialcorr'
