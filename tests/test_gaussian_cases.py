# -*- coding: utf-8 -*-
"""Gaussian-likelihood family: gaussian_nll / kl_diagonal_gaussian.

Pins the contract + that nitrix matches the exact-formula numpy fp64 oracle
(and, for the NLL, agrees with the scipy.stats community cross-check).
"""
import jax
import numpy as np
import pytest

from nperf.measure import load_case

_EXPECT = {
    'gaussian_nll': {'nitrix-jax', 'numpy.gaussian_nll', 'scipy.norm_nll',
                     'cupy.gaussian_nll'},
    'kl_diagonal_gaussian': {'nitrix-jax', 'numpy.kl_diagonal_gaussian',
                             'cupy.kl_diagonal_gaussian'},
}


@pytest.mark.parametrize('name', list(_EXPECT))
def test_contract_and_oracle(name):
    c = load_case(name)
    bp = c.build({'shape': [1024, 32], 'seed': 0})
    assert set(bp.baselines) == _EXPECT[name]
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname.startswith('nitrix.stats.')
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    ref = np.asarray(bp.fp64_reference)
    rel = np.max(np.abs(out - ref) / (c.atol + c.rtol * np.abs(ref)))
    assert rel <= 1.0, f'{name}: rel_to_tol {rel:.2f} > 1 (vs oracle)'


def test_nll_matches_scipy_community():
    '''gaussian_nll's scipy.stats cross-check agrees with the exact formula --
    the two community impls coincide (so the community bar is sound).'''
    c = load_case('gaussian_nll')
    bp = c.build({'shape': [512, 16], 'seed': 1})
    numpy_ref = np.asarray(bp.baselines['numpy.gaussian_nll'][1](
        *bp.inputs_for('numpy')))
    scipy_ref = np.asarray(bp.baselines['scipy.norm_nll'][1](
        *bp.inputs_for('numpy')))
    assert np.allclose(numpy_ref, scipy_ref, rtol=1e-5, atol=1e-6)
