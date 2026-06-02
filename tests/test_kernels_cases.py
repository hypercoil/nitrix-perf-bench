# -*- coding: utf-8 -*-
"""Tier-2 kernel cases (rbf_kernel / linear_kernel / linear_distance).

CPU build + oracle agreement for the runnable (jax + sklearn) baselines; the
cupy GPU ref is skipped here (needs a device + the refs env).  These ops are
matmul/broadcast based, so -- unlike the eigh family -- the cupy ref is *not*
solver-blocked: it is skipped only for lack of a device, not on principle.
"""
import pytest

from nperf.cases import linear_distance, linear_kernel, rbf_kernel
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'n': 64, 'd': 8, 'seed': 0}
_CASES = [
    (rbf_kernel, 'sklearn.rbf_kernel'),
    (linear_kernel, 'sklearn.linear_kernel'),
    (linear_distance, 'sklearn.euclidean_distances'),
]


@pytest.mark.parametrize('mod,refname', _CASES)
def test_baseline_shape(mod, refname):
    built = mod._build(_P)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'  # cupy ref


@pytest.mark.parametrize('mod,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, refname):
    built = mod._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        args = built.inputs_for(framework_of(provider_id))
        fid = compare(fn(*args), built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_oracle_shape_is_square():
    built = rbf_kernel._build(_P)
    assert built.fp64_reference.shape == (_P['n'], _P['n'])


def test_op_qualnames_match_nitrix():
    assert rbf_kernel.CASE.op_qualname == 'nitrix.linalg.rbf_kernel'
    assert linear_kernel.CASE.op_qualname == 'nitrix.linalg.linear_kernel'
    assert (linear_distance.CASE.op_qualname
            == 'nitrix.linalg.linear_distance')
