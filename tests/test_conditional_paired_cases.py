# -*- coding: utf-8 -*-
"""Tier-2 paired / conditional family cases.

CPU build + oracle agreement for the runnable (jax + numpy) baselines; the cupy
GPU twin is skipped here (needs a device + the refs env).  ``pairedcov`` /
``pairedcorr`` are pure-BLAS cross-blocks; the ``conditional*`` pair
residualise ``X`` against ``Y`` (a tiny ``(d, d)`` Gram solve) then cov.
"""
import numpy as np
import pytest

from nperf.cases import (
    conditionalcorr,
    conditionalcov,
    pairedcorr,
    pairedcov,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# small CPU points (one per case): paired use a square cross-block (c = d);
# conditional use d << c (a handful of confounds), the realistic framing.
_CASES = [
    (pairedcov, {'c': 32, 'd': 32, 'obs': 256, 'seed': 0}, 'numpy.pairedcov'),
    (pairedcorr,
     {'c': 32, 'd': 32, 'obs': 256, 'seed': 0}, 'numpy.pairedcorr'),
    (conditionalcov,
     {'c': 32, 'd': 8, 'obs': 256, 'seed': 0}, 'numpy.conditionalcov'),
    (conditionalcorr,
     {'c': 32, 'd': 8, 'obs': 256, 'seed': 0}, 'numpy.conditionalcorr'),
]


@pytest.mark.parametrize('mod,p,refname', _CASES)
def test_baseline_shape(mod, p, refname):
    built = mod._build(p)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    # the only extra is the cupy GPU on-target twin.
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'


@pytest.mark.parametrize('mod,p,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, p, refname):
    built = mod._build(p)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU twin: needs a device + the refs env
        args = built.inputs_for(framework_of(provider_id))
        fid = compare(np.asarray(fn(*args)), built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_paired_output_is_cross_block():
    '''pairedcov / pairedcorr return the (c, d) cross-block (not square).'''
    cov = pairedcov._build({'c': 12, 'd': 5, 'obs': 128, 'seed': 0})
    assert np.asarray(cov.fp64_reference).shape == (12, 5)
    corr = pairedcorr._build({'c': 12, 'd': 5, 'obs': 128, 'seed': 0})
    r = np.asarray(corr.fp64_reference)
    assert r.shape == (12, 5)
    assert np.all(np.abs(r) <= 1.0 + 1e-6)  # correlations are bounded


def test_conditioning_moves_the_covariance():
    '''The shared-factor input makes the conditioning *matter*: residualising
    Y out changes the covariance vs plain cov(X) (otherwise the case would be a
    trivial restatement of cov). Pins that the residualisation is exercised.'''
    from nitrix.stats import cov as nitrix_cov

    from nperf.cases._conditional_paired import (
        paired_conditional,
        paired_input,
    )
    X, Y = paired_input(32, 8, 256, seed=0)
    ccov = paired_conditional(X.astype(np.float64), Y.astype(np.float64),
                              'conditionalcov', np)
    plain = np.asarray(nitrix_cov(X)).astype(np.float64)
    rel = np.linalg.norm(ccov - plain) / np.linalg.norm(plain)
    assert rel > 0.1, f'conditioning barely moved the cov (rel {rel:.3g})'


def test_conditionalcorr_unit_diagonal():
    '''conditionalcorr is a correlation matrix: ~unit diagonal, entries in
    [-1, 1].'''
    built = conditionalcorr._build({'c': 24, 'd': 8, 'obs': 256, 'seed': 0})
    r = np.asarray(built.fp64_reference)
    assert np.allclose(np.diag(r), 1.0, atol=1e-6)
    assert np.all(np.abs(r) <= 1.0 + 1e-6)


def test_op_qualnames_match_nitrix():
    assert pairedcov.CASE.op_qualname == 'nitrix.stats.pairedcov'
    assert pairedcorr.CASE.op_qualname == 'nitrix.stats.pairedcorr'
    assert conditionalcov.CASE.op_qualname == 'nitrix.stats.conditionalcov'
    assert (conditionalcorr.CASE.op_qualname
            == 'nitrix.stats.conditionalcorr')
