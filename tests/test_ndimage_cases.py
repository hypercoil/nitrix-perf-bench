# -*- coding: utf-8 -*-
"""PERF_AUDIT scipy.ndimage ports (B11 slice 2): gaussian / erode / dilate.

The cleanly-comparable ndimage ops: nitrix (jax) vs scipy.ndimage (host), which
agree to round-off (erode/dilate exactly), so scipy-in-fp64 is the oracle and
both baselines pass.  scipy is in the base env, so both run in the unit env.
(The divergent ndimage ops -- distance_transform / median_filter /
spatial_transform -- are a separate slice: they need a no-cross-impl-oracle
fidelity path, not a forced scipy oracle.)
"""
import numpy as np
import pytest

from nperf.cases import dilate, erode, gaussian
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_CASES = [
    (gaussian, {'shape': [32, 32], 'sigma': 1.5, 'seed': 0},
     'scipy.ndimage.gaussian_filter'),
    (erode, {'shape': [32, 32], 'size': 3, 'seed': 0},
     'scipy.ndimage.grey_erosion'),
    (dilate, {'shape': [32, 32], 'size': 3, 'seed': 0},
     'scipy.ndimage.grey_dilation'),
]


def test_scipy_provider_is_numpy_framework():
    assert framework_of('scipy') == 'numpy'


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_baseline_shape(mod, param, refname):
    built = mod._build(param)
    assert set(built.baselines) == {'nitrix-jax', refname}
    assert built.baselines[refname][0] == 'scipy'
    assert built.ratio_reference == 'nitrix-jax'


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_both_baselines_match_oracle(mod, param, refname):
    built = mod._build(param)
    for name, (framework, fn) in built.baselines.items():
        out = np.asarray(fn(*built.inputs_for(framework)), dtype=np.float64)
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames_match_nitrix():
    assert gaussian.CASE.op_qualname == 'nitrix.smoothing.gaussian'
    assert erode.CASE.op_qualname == 'nitrix.morphology.erode'
    assert dilate.CASE.op_qualname == 'nitrix.morphology.dilate'
