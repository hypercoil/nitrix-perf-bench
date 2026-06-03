# -*- coding: utf-8 -*-
"""Tier-2 SPD eigh-family cases (symlog / symsqrt / sympower).

CPU build + oracle agreement for the runnable (jax + scipy) baselines on a
well-conditioned SPD input.  The cupy GPU reference is skipped here (it needs a
device + the refs env); the GPU path is blocked upstream (the jaxlib cuSOLVER
bug) and is exercised by the runner, not the unit suite.
"""
import numpy as np
import pytest

from nperf.cases import symexp, symlog, sympower, symsqrt
from nperf.core.fidelity import compare
from nperf.providers import requires_of

_CASES = [
    (symlog, {'d': 32, 'seed': 0}, 'scipy.linalg.logm'),
    (symexp, {'d': 32, 'seed': 0}, 'scipy.linalg.expm'),
    (symsqrt, {'d': 32, 'seed': 0}, 'scipy.linalg.sqrtm'),
    (sympower, {'d': 32, 'power': 0.75, 'seed': 0},
     'scipy.linalg.fractional_matrix_power'),
]


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_baseline_shape(mod, param, refname):
    built = mod._build(param)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.baselines['nitrix-jax'][0] == 'jax'
    assert built.ratio_reference == 'nitrix-jax'
    # the extra baseline is the GPU-only on-target ref (cupy, eigh-based).
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, param, refname):
    built = mod._build(param)
    assert built.fp64_reference is not None
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(provider_id)), dtype=np.float64)
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames_match_nitrix():
    assert symlog.CASE.op_qualname == 'nitrix.linalg.symlog'
    assert symexp.CASE.op_qualname == 'nitrix.linalg.symexp'
    assert symsqrt.CASE.op_qualname == 'nitrix.linalg.symsqrt'
    assert sympower.CASE.op_qualname == 'nitrix.linalg.sympower'
