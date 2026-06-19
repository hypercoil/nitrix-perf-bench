# -*- coding: utf-8 -*-
"""Permutation-inference family (nitrix.stats.inference.*).

Pins the contract + that nitrix (and the exact CPU community baselines) agree
with the fp64 oracle on small points. cupy (GPU) and FSL (subprocess) baselines
are validated in the measurement run, not here -- this stays CPU + base-venv.

Covers the 6 atomic ops (tfce / cluster_size_map / cluster_mass_map / fdr_bh /
bonferroni / gpd_pvalue) AND the cluster-extent CHAIN (the chain-parity case:
nitrix's fused supra_threshold_clusters->cluster_size_map vs FSL fsl-cluster).
"""
import jax
import numpy as np
import pytest

from nperf.measure import load_case

# case -> (small build param, expected baseline names)
_CASES = {
    'tfce': ({'shape': [24, 24, 24], 'seed': 0},
             {'nitrix-jax', 'cupyx.ndimage.tfce'}),
    'cluster_size_map': ({'shape': [24, 24, 24], 'seed': 0},
                         {'nitrix-jax', 'cupyx.ndimage.cluster_size'}),
    'cluster_mass_map': ({'shape': [24, 24, 24], 'seed': 0},
                         {'nitrix-jax', 'cupyx.ndimage.cluster_mass'}),
    'cluster_extent': ({'shape': [24, 24, 24], 'seed': 0},
                       {'nitrix-jax', 'fsl.fsl-cluster', 'fsl.iofloor'}),
    'fdr_bh': ({'n': 5000, 'seed': 0},
               {'nitrix-jax', 'statsmodels.multipletests'}),
    'bonferroni': ({'n': 5000, 'seed': 0},
                   {'nitrix-jax', 'statsmodels.multipletests'}),
    'gpd_pvalue': ({'n': 5000, 'seed': 0},
                   {'nitrix-jax', 'scipy.genpareto'}),
}

# baselines compared to the oracle here: exact + CPU + base-venv-importable.
# (cupy = GPU; fsl = subprocess; scipy.genpareto = Approx -- all skipped here)
_GATED = {'jax', 'statsmodels'}


@pytest.mark.parametrize('name', list(_CASES))
def test_contract(name):
    param, expect = _CASES[name]
    c = load_case(name)
    bp = c.build(param)
    assert set(bp.baselines) == expect
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname.startswith('nitrix.stats.inference')


@pytest.mark.parametrize('name', list(_CASES))
def test_exact_baselines_match_oracle(name):
    param, _ = _CASES[name]
    c = load_case(name)
    bp = c.build(param)
    ref = np.asarray(bp.fp64_reference)
    checked = 0
    for bn, (fw, fn) in bp.baselines.items():
        if fw not in _GATED:
            continue
        out = np.asarray(jax.block_until_ready(fn(*bp.inputs_for(fw))))
        rel = np.max(np.abs(out - ref) / (c.atol + c.rtol * np.abs(ref)))
        assert rel <= 1.0, f'{name}/{bn}: rel_to_tol {rel:.2f} > 1'
        checked += 1
    assert checked >= 1  # nitrix-jax at minimum


def test_cluster_extent_is_a_chain():
    '''The chain case fuses supra_threshold_clusters -> cluster_size_map and
    targets the labelling op (no atomic community parity); fsl.iofloor is an
    approximate (no-op) baseline.'''
    c = load_case('cluster_extent')
    assert c.op_qualname == \
        'nitrix.stats.inference.cluster.supra_threshold_clusters'
    assert any(a.baseline == 'fsl.iofloor' for a in c.approximate_baselines)
    assert any(s.baseline == 'fsl.fsl-cluster' for s in c.slow_baselines)


def test_gpd_scipy_is_approximate():
    '''scipy genpareto (MLE) is a reported-not-gated cross-check of nitrix's
    method-of-moments GPD.'''
    c = load_case('gpd_pvalue')
    assert any(a.baseline == 'scipy.genpareto'
               for a in c.approximate_baselines)


@pytest.mark.parametrize('name', ['fdr_bh', 'bonferroni'])
def test_padjusted_is_a_valid_pvalue(name):
    '''Adjusted p-values stay in [0, 1] and never fall below the raw p.'''
    c = load_case(name)
    bp = c.build({'n': 5000, 'seed': 0})
    from nperf.cases._inference import pvalues
    raw = pvalues(5000, 0)
    padj = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    assert padj.min() >= 0.0 and padj.max() <= 1.0 + 1e-6
    assert np.all(padj >= np.asarray(raw) - 1e-5)  # correction only inflates


def test_tfce_and_clusters_nonnegative():
    '''TFCE enhancement and cluster size/mass maps are non-negative.'''
    for name in ('tfce', 'cluster_size_map', 'cluster_mass_map'):
        c = load_case(name)
        bp = c.build({'shape': [24, 24, 24], 'seed': 0})
        out = np.asarray(jax.block_until_ready(
            bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
        assert out.min() >= -1e-6, f'{name} has negative values'


# -- permutation_test HEADLINE (separate: Monte Carlo, fp64_reference=None) ---
def test_permutation_test_contract():
    c = load_case('permutation_test')
    bp = c.build({'shape': [16, 16, 16], 'subj': 12, 'n_perm': 50, 'seed': 0})
    assert set(bp.baselines) == {
        'nitrix-jax', 'fsl.randomise', 'fsl.iofloor', 'nilearn.permuted_ols'}
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname == 'nitrix.stats.inference.permutation_test'
    assert c.tier == 'marquee'
    # permutation FWE p-maps are Monte Carlo -> no exact oracle, but noted.
    assert bp.fp64_reference is None
    assert bp.fidelity_note and 'Monte Carlo' in bp.fidelity_note
    # both community permutation tools are declared slow (CPU loops).
    slow = {s.baseline for s in c.slow_baselines}
    assert {'fsl.randomise', 'nilearn.permuted_ols'} <= slow


def test_permutation_test_stat_matches_oracle():
    '''The DETERMINISTIC observed t-map (PermResult.stat) matches a numpy one-
    sample-t oracle, even though p_fwe is Monte Carlo.'''
    from nitrix.stats.inference import permutation_test

    from nperf.cases._inference import np_onesample_t, perm_data
    shape, subj = (16, 16, 16), 12
    data, design, contrast = perm_data(shape, subj, 0)
    r = permutation_test(
        jax.numpy.asarray(data), jax.numpy.asarray(design),
        jax.numpy.asarray(contrast), key=jax.random.PRNGKey(0),
        n_perm=50, enhancement='tfce')
    stat = np.asarray(jax.block_until_ready(r.stat))
    ref = np_onesample_t()(data)
    rel = np.max(np.abs(stat - ref) / (1e-4 + 1e-3 * np.abs(ref)))
    assert rel <= 1.0, f'observed stat vs numpy t: rel_to_tol {rel:.2f} > 1'


def test_permutation_test_recovers_planted_cluster():
    '''p_fwe is a valid p-map and the planted central cube is more significant
    than the background (the effect is detected).'''
    c = load_case('permutation_test')
    shape = [16, 16, 16]
    bp = c.build({'shape': shape, 'subj': 12, 'n_perm': 100, 'seed': 0})
    p = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    assert p.min() >= 0.0 and p.max() <= 1.0 + 1e-6
    blob = np.zeros(shape, bool)
    sl = tuple(slice(s // 3, 2 * s // 3) for s in shape)
    blob[tuple(sl)] = True
    assert np.median(p[blob]) < np.median(p[~blob])
