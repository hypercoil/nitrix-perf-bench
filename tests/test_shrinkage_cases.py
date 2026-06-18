# -*- coding: utf-8 -*-
"""Shrinkage-covariance + sparse-precision family (nitrix.stats.connectivity).

Pins (a) the case contract, (b) that the nitrix op matches its fp64 oracle
within the declared tolerance -- here the oracle IS the strong community
baseline (scikit-learn is the reference impl of Ledoit-Wolf / OAS / graphical
LASSO), so a passing fidelity check is also cross-tool agreement -- and (c) the
structural properties (SPD cov, symmetric/sparse precision).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nperf.measure import load_case

_CASES = ['ledoit_wolf', 'oas', 'shrunk_covariance', 'glasso', 'glasso_path',
          'ebic_score']
_COMMUNITY = {  # the strong baseline each case carries besides nitrix-jax
    'ledoit_wolf': 'sklearn.ledoit_wolf',
    'oas': 'sklearn.oas',
    'shrunk_covariance': 'sklearn.ledoit_wolf',
    'glasso': 'sklearn.graphical_lasso',
    'glasso_path': 'sklearn.graphical_lasso_path',
    'ebic_score': 'numpy.ebic_score',
}


@pytest.mark.parametrize('name', _CASES)
def test_contract(name):
    c = load_case(name)
    bp = c.build(c.representative)
    assert set(bp.baselines) == {'nitrix-jax', _COMMUNITY[name]}
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname.startswith('nitrix.stats.')
    assert bp.fp64_reference is not None  # all 6 carry a numeric oracle
    assert c.large_param_points  # a brain-scale size tier


@pytest.mark.parametrize('name', _CASES)
def test_matches_oracle(name):
    '''nitrix matches the fp64 oracle (= the sklearn/numpy reference) to the
    declared tolerance -- the accuracy pin AND cross-tool agreement.'''
    c = load_case(name)
    bp = c.build(c.representative)
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    ref = np.asarray(bp.fp64_reference)
    rel = np.max(np.abs(out - ref) / (c.atol + c.rtol * np.abs(ref)))
    assert rel <= 1.0, f'{name}: rel_to_tol {rel:.2f} > 1 (out vs oracle)'


def test_shrinkage_cov_is_spd():
    '''Ledoit-Wolf shrinks toward a scaled identity -> a well-conditioned SPD
    covariance (the point of shrinkage in the p>n regime).'''
    from nitrix.stats import ledoit_wolf

    from nperf.cases._shrinkage import shrinkage_data
    cov = np.asarray(ledoit_wolf(jnp.asarray(shrinkage_data(60, 200, 0)))[0])
    eig = np.linalg.eigvalsh(cov)
    assert eig.min() > 0, 'shrunk covariance not SPD'
    assert np.allclose(cov, cov.T, atol=1e-5)


def test_glasso_precision_sparse_symmetric():
    '''Graphical-LASSO precision is symmetric and genuinely sparse (the L1
    penalty zeros off-diagonal edges) -- and SPD.'''
    from nitrix.stats import glasso

    from nperf.cases._shrinkage import sparse_precision_cov
    S = sparse_precision_cov(60, 0)
    theta = np.asarray(glasso(jnp.asarray(S), 0.15))
    assert np.allclose(theta, theta.T, atol=1e-5)
    assert np.linalg.eigvalsh(theta).min() > 0, 'precision not SPD'
    off = theta[~np.eye(60, dtype=bool)]
    assert np.mean(np.abs(off) < 1e-6) > 0.2, 'precision not sparse'
