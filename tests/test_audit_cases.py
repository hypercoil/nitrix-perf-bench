# -*- coding: utf-8 -*-
"""PERF_AUDIT-port cases (B11): cov / corr / residualise vs numpy refs.

Each pairs a nitrix op (jax) with the natural numpy reference, scored against
an fp64 oracle.  The CPU baselines (jax + numpy) are pure jax+numpy, so unlike
the torch/PyG cases they run fully in the unit env -- the test exercises them
for real.  A GPU-only on-target ref (cupy, Phase B), where a case declares one,
is *present* but not run here: it needs a device + the refs-cupy env, so it is
skipped (the runner records ``platform_not_applicable`` off-GPU).
"""
import numpy as np
import pytest

from nperf.cases import corr, cov, residualise
from nperf.core.fidelity import compare
from nperf.providers import requires_of

# (module, small param point, expected reference baseline name)
_CASES = [
    (cov, {'c': 16, 'n_obs': 64, 'seed': 0}, 'numpy.cov'),
    (corr, {'n': 16, 't': 64, 'seed': 0}, 'numpy.corrcoef'),
    (residualise, {'V': 128, 'N': 40, 'K': 6, 'seed': 0},
     'numpy.linalg.lstsq'),
]


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_baseline_shape(mod, param, refname):
    built = mod._build(param)
    names = set(built.baselines)
    # the jax SUT + the numpy CPU reference are always present...
    assert {'nitrix-jax', refname} <= names
    assert built.baselines[refname][0] == 'numpy'      # numpy provider
    assert built.baselines['nitrix-jax'][0] == 'jax'
    assert built.ratio_reference == 'nitrix-jax'
    # ...and any extra baseline is a GPU-only on-target ref (e.g. cupy).
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_both_baselines_match_oracle(mod, param, refname):
    built = mod._build(param)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # GPU-only ref (cupy): needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(provider_id)), dtype=np.float64)
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames_match_nitrix():
    assert cov.CASE.op_qualname == 'nitrix.stats.cov'
    assert corr.CASE.op_qualname == 'nitrix.stats.corr'
    assert residualise.CASE.op_qualname == 'nitrix.linalg.residualise'
