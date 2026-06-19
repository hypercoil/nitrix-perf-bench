# -*- coding: utf-8 -*-
"""GAM/spline family (nitrix.stats.gam.gam_fit) vs R mgcv::gam.

The headline mass-univariate penalised-spline GAM. Fidelity is mgcv agreement
(Monte-Carlo-free but no exact cross-tool oracle), so the case carries
fp64_reference=None; here we pin (a) the contract, (b) truth recovery, (c) the
penalised-IRLS CORE against a numpy fixed-lambda oracle, and (d) -- guarded by
Rscript -- agreement with mgcv.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nperf.cases import gam_fit
from nperf.cases._gam import gam_data, np_penalised_ls

_RSCRIPT = os.environ.get('NPERF_RSCRIPT', '/scratch/nperf/renv/bin/Rscript')
_P = {'V': 32, 'N': 150, 'n_basis': 15, 'seed': 0}


def test_contract():
    built = gam_fit._build(_P)
    assert set(built.baselines) == {'nitrix-jax', 'R.mgcv', 'R.iofloor'}
    assert built.ratio_reference == 'nitrix-jax'
    assert gam_fit.CASE.op_qualname == 'nitrix.stats.gam.gam_fit'
    assert gam_fit.CASE.tier == 'marquee'
    assert built.fp64_reference is None and built.fidelity_note
    slow = {s.baseline for s in gam_fit.CASE.slow_baselines}
    assert 'R.mgcv' in slow
    assert any(a.baseline == 'R.iofloor'
               for a in gam_fit.CASE.approximate_baselines)


def test_recovers_true_smooth():
    '''The GAM (selection) recovers the planted smooth f(x) = sin(2 pi x).'''
    built = gam_fit._build(_P)
    yhat = np.asarray(jax.block_until_ready(
        built.baselines['nitrix-jax'][1](*built.inputs_for('jax'))))
    x, _ = gam_data(_P['V'], _P['N'], _P['seed'])
    f = np.sin(2.0 * np.pi * x)
    assert np.corrcoef(yhat.mean(0), f)[0, 1] > 0.99


def test_irls_core_matches_fixed_lambda_oracle():
    '''The penalised-IRLS core is exact: at a FIXED lambda, nitrix's fitted
    smooth matches the numpy penalised-least-squares oracle.'''
    from nitrix.stats.basis import bspline_basis
    from nitrix.stats.gam import gam_fit as gf
    x, Y = gam_data(8, 150, 0)
    sb = bspline_basis(jnp.asarray(x), 15, center=True)
    big_d = jnp.concatenate(
        [jnp.ones((150, 1), sb.design.dtype), sb.design], axis=1)
    lam = 1.0
    res = gf(jnp.asarray(Y), [sb], lam_floor=lam, lam_ceil=lam)
    yhat = np.asarray((big_d @ res.coef.T).T)
    s_full = np.zeros((big_d.shape[1],) * 2)
    s_full[1:, 1:] = np.asarray(sb.penalty, np.float64)
    ref = np_penalised_ls(np.asarray(big_d), s_full, lam)(Y)
    rel = np.max(np.abs(yhat - ref) / (1e-4 + 1e-3 * np.abs(ref)))
    assert rel <= 1.0, f'IRLS core vs fixed-lambda oracle: rel {rel:.3g}'


@pytest.mark.skipif(not os.path.exists(_RSCRIPT), reason='Rscript absent')
def test_agrees_with_mgcv():
    '''nitrix and R mgcv (both P-spline + REML) produce essentially identical
    fitted smooths (corr > 0.999).'''
    built = gam_fit._build(_P)
    yn = np.asarray(jax.block_until_ready(
        built.baselines['nitrix-jax'][1](*built.inputs_for('jax'))))
    Y, x = built.inputs_for('numpy')
    ym = built.baselines['R.mgcv'][1](np.asarray(Y)[:8], x)  # subset: quick
    assert np.corrcoef(yn[:8].ravel(), ym.ravel())[0, 1] > 0.999


# -- spline-type breadth (cyclic / thinplate / tensor) -----------------------
_BREADTH = {
    'cyclic_gam': 'nitrix.stats.basis.cyclic_cubic_basis',
    'thinplate_gam': 'nitrix.stats.basis.thinplate_regression_basis',
    'tensor_gam': 'nitrix.stats.basis.tensor_product_basis',
}


@pytest.mark.parametrize('name', list(_BREADTH))
def test_breadth_contract(name):
    from nperf.measure import load_case
    c = load_case(name)
    bp = c.build(c.representative)
    assert set(bp.baselines) == {'nitrix-jax', 'R.mgcv', 'R.iofloor'}
    assert c.op_qualname == _BREADTH[name]
    assert bp.fp64_reference is None and bp.fidelity_note
    assert any(s.baseline == 'R.mgcv' for s in c.slow_baselines)


@pytest.mark.skipif(not os.path.exists(_RSCRIPT), reason='Rscript absent')
@pytest.mark.parametrize('name,floor', [
    ('cyclic_gam', 0.99), ('thinplate_gam', 0.99), ('tensor_gam', 0.97)])
def test_breadth_agrees_with_mgcv(name, floor):
    '''Each smooth type's fitted curve/surface agrees with the matching mgcv
    smooth (cc / tp / te).'''
    from nperf.measure import load_case
    c = load_case(name)
    bp = c.build(c.representative)
    yn = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    Y, X = bp.inputs_for('numpy')
    ym = bp.baselines['R.mgcv'][1](np.asarray(Y)[:8], X)
    assert np.corrcoef(yn[:8].ravel(), ym.ravel())[0, 1] > floor
