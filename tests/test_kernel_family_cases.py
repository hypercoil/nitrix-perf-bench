# -*- coding: utf-8 -*-
"""Tier-2 kernel-family cases (gaussian / cosine / polynomial / sigmoid).

CPU build + oracle agreement for the runnable (jax + sklearn) baselines; the
cupy GPU ref is skipped here (needs a device + the refs env). Each nitrix
kernel is matched to its ``sklearn.metrics.pairwise`` equivalent with
translated parameters (order->degree, r->coef0, gaussian
sigma->gamma=1/(2*sigma^2)), so the sklearn agreement is the right-target
check.
"""
import numpy as np
import pytest

from nperf.cases import (
    cosine_kernel,
    gaussian_kernel,
    polynomial_kernel,
    sigmoid_kernel,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_MODS = [gaussian_kernel, cosine_kernel, polynomial_kernel, sigmoid_kernel]
_P = {'n': 64, 'd': 32, 'seed': 0}


@pytest.mark.parametrize('mod', _MODS)
def test_baselines(mod):
    built = mod._build(_P)
    names = set(built.baselines)
    assert 'nitrix-jax' in names and built.ratio_reference == 'nitrix-jax'
    assert any(n.startswith('sklearn.') for n in names)  # CPU floor
    gpu = [n for n in names if n.startswith('cupy.')]
    assert len(gpu) == 1 and requires_of(built.baselines[gpu[0]][0]) == 'gpu'


@pytest.mark.parametrize('mod', _MODS)
def test_host_baselines_match_sklearn_oracle(mod):
    built = mod._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy ref: needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames():
    assert gaussian_kernel.CASE.op_qualname == 'nitrix.linalg.gaussian_kernel'
    assert cosine_kernel.CASE.op_qualname == 'nitrix.linalg.cosine_kernel'
    assert (polynomial_kernel.CASE.op_qualname
            == 'nitrix.linalg.polynomial_kernel')
    assert sigmoid_kernel.CASE.op_qualname == 'nitrix.linalg.sigmoid_kernel'
