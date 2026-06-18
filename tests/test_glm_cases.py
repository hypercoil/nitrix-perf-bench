# -*- coding: utf-8 -*-
"""Mass-univariate GLM family (nitrix.stats.glm_fit / t_contrast / f_contrast).

Pins the contract + that nitrix AND both community tools (nilearn vectorised,
statsmodels per-voxel loop) agree with the exact numpy OLS fp64 oracle -- so a
passing check is genuine cross-tool agreement on the betas / t / F.
"""
import jax
import numpy as np
import pytest

from nperf.measure import load_case

_EXPECT = {
    'glm_fit': {'nitrix-jax', 'nilearn.run_glm', 'statsmodels.OLS'},
    't_contrast': {'nitrix-jax', 'nilearn.compute_contrast',
                   'statsmodels.t_test'},
    'f_contrast': {'nitrix-jax', 'nilearn.compute_contrast',
                   'statsmodels.f_test'},
}


@pytest.mark.parametrize('name', list(_EXPECT))
def test_contract_and_all_baselines_agree(name):
    c = load_case(name)
    bp = c.build({'V': 200, 'N': 80, 'p': 6, 'seed': 0})
    assert set(bp.baselines) == _EXPECT[name]
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname.startswith('nitrix.stats.')
    # statsmodels (the loop) is declared slow on every GLM case.
    assert any('statsmodels' in s.baseline for s in c.slow_baselines)
    ref = np.asarray(bp.fp64_reference)
    for bn, (fw, fn) in bp.baselines.items():
        out = np.asarray(jax.block_until_ready(fn(*bp.inputs_for(fw))))
        rel = np.max(np.abs(out - ref) / (c.atol + c.rtol * np.abs(ref)))
        assert rel <= 1.0, f'{name}/{bn}: rel_to_tol {rel:.2f} > 1'


def test_glm_fit_coef_shape():
    '''glm_fit returns per-voxel betas coef[V, p] (V voxels, p regressors).'''
    c = load_case('glm_fit')
    bp = c.build({'V': 128, 'N': 60, 'p': 5, 'seed': 0})
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    assert out.shape == (128, 5)
