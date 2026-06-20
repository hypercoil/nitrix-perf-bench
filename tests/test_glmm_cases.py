# -*- coding: utf-8 -*-
"""GLMM marquee (``nitrix.stats.glmm.glmm_fit``): contract + recovery + the v3
no-divergence guard, across the family x structure x method modelling paths.

The Gaussian random intercept IS the LME, gated against the closed-form
balanced REML oracle (exact).  The non-Gaussian / slope paths have no closed
form: we pin (a) the path / contract shape, (b) that the FIXED effects recover
the planted truth, (c) -- the v3 regression guard -- that the numerically-hard
cells (the robust unstructured slope, binary AGQ) stay FINITE (no divergence /
overflow / indefinite-Hessian blow-up), and (d) -- guarded by Rscript --
agreement with the R mgcv re-smooth PQL reference.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.stats.glmm import glmm_fit

from nperf.cases import glmm_fit as G
from nperf.cases._glmm import glmm_data, r_mgcv_glmm
from nperf.report.sizing import label, size_elems

_RSCRIPT = os.environ.get('NPERF_RSCRIPT', '/scratch/nperf/renv/bin/Rscript')


def _first(path):
    return next(p for p in G.CASE.param_points if p['path'] == path)


def _run_nitrix(param):
    bp = G._build(param)
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    return bp, out


def test_paths_and_axes():
    """The case declares the 5-cell config space, both scale axes, the two
    challenging paths, and path-labelled anchors."""
    labels = [sp.label for sp in G.CASE.scale_paths]
    assert len(labels) == 5 and len(set(labels)) == 5
    challenging = {sp.label for sp in G.CASE.scale_paths if sp.challenging}
    assert challenging == {'gaussian-slope-unstructured', 'binomial-slope-agq'}
    assert {sp.cost.axis for sp in G.CASE.scale_paths} == {'V', 'q'}
    assert len(G.CASE.param_points) == 24      # 5+5+5+5+4 per-path grids
    assert len(G.CASE.large_param_points) == 2
    for a in G.CASE.large_param_points:
        assert a['path'] in labels        # each anchor extends a real path


def test_contract_baselines_per_path():
    """Non-Gaussian intercept paths carry the looped R baseline + I/O floor;
    the Gaussian (oracle-gated) and slope / AGQ (nitrix-only) paths do not."""
    assert set(G._build(_first('gaussian-intercept-few')).baselines) == {
        'nitrix-jax'}
    for path in ('binomial-intercept-many', 'poisson-intercept-few'):
        assert set(G._build(_first(path)).baselines) == {
            'nitrix-jax', 'R.mgcv', 'R.iofloor'}
    for path in ('gaussian-slope-unstructured', 'binomial-slope-agq'):
        assert set(G._build(_first(path)).baselines) == {'nitrix-jax'}
        assert G._build(_first(path)).fp64_reference is None


def test_gaussian_intercept_matches_oracle():
    """Gaussian random intercept == LME -> gated vs the closed-form REML."""
    bp, out = _run_nitrix(G._REPRESENTATIVE)
    ref = np.asarray(bp.fp64_reference)
    assert out.shape == ref.shape == (64, 3)
    rel = np.max(np.abs(out - ref) / (G.CASE.atol + G.CASE.rtol * np.abs(ref)))
    assert rel <= 1.0, f'rel_to_tol {rel:.2f} > 1 vs the closed-form oracle'


def test_nonlinear_intercept_recovers_planted():
    """Binomial / Poisson random-intercept fixed effects recover the planted
    intercept (beta0=0.3) and stay finite (PQL attenuates the variance, not the
    fixed effect)."""
    for path in ('binomial-intercept-many', 'poisson-intercept-few'):
        _, out = _run_nitrix(_first(path))
        assert np.isfinite(out).all()
        assert abs(float(out[:, 0].mean()) - 0.3) < 0.25


def test_challenging_slope_paths_stay_finite():
    """The v3 regression guard: the robust unstructured-slope solver and binary
    AGQ produce FINITE fixed effects AND finite variance components (no
    catastrophic divergence / overflow / indefinite-Hessian blow-up), and the
    fixed effects recover the planted [0.3, 0.5]."""
    cells = [('unstructured', 'pql', 'gaussian', 5),
             ('unstructured', 'agq', 'binomial', 5)]
    for structure, method, fam, nq in cells:
        Y, X, group, z, truth = glmm_data(fam, 24, 16, 12, structure, seed=0)
        r = glmm_fit(jnp.asarray(Y), jnp.asarray(X), group=jnp.asarray(group),
                     z=jnp.asarray(z), structure=structure, family=fam,
                     method=method, n_quad=nq)
        beta = np.asarray(r.beta_hat)
        assert np.isfinite(beta).all(), f'{fam}/{method}: non-finite beta'
        assert np.isfinite(np.asarray(r.re_var)).all(), \
            f'{fam}/{method}: non-finite variance component (divergence)'
        assert abs(float(beta[:, 0].mean()) - truth['beta0']) < 0.3
        assert abs(float(beta[:, 1].mean()) - truth['beta1']) < 0.3


def test_sizing_glmm():
    """``size_elems`` is the V*q ordering proxy; ``label`` shows the path."""
    p = _first('binomial-intercept-many')
    assert size_elems(p) == int(p['V']) * int(p['q'])
    lbl = label(p)
    assert 'bin' in lbl and 'int' in lbl and 'pql' in lbl
    assert 'nq5' in label(_first('binomial-slope-agq'))


@pytest.mark.skipif(not os.path.exists(_RSCRIPT), reason='no Rscript')
def test_r_mgcv_agreement():
    """nitrix binomial PQL agrees with the R mgcv s(g,bs="re") estimator it
    targets (the intercept correlates across voxels)."""
    Y, X, group, z, truth = glmm_data('binomial', 16, 12, 10, 'intercept',
                                      seed=1)
    r = glmm_fit(jnp.asarray(Y), jnp.asarray(X), group=jnp.asarray(group),
                 family='binomial', method='pql')
    nit = np.asarray(r.beta_hat)[:, 0]
    rref = r_mgcv_glmm(Y, group, 'binomial')
    assert np.isfinite(rref).all()
    assert np.corrcoef(nit, rref[:, 0])[0, 1] > 0.9
