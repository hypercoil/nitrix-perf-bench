# -*- coding: utf-8 -*-
"""Tier-2 PCA family cases (pca_fit / pca_transform / pca_inverse_transform).

CPU build + oracle agreement for the runnable (jax + numpy/sklearn) baselines;
the cupy GPU ref is skipped here (needs a device + the refs env).  ``pca_fit``
scores the sign/rotation-invariant ``explained_variance`` (top-``k`` covariance
eigenvalues); ``pca_transform`` / ``pca_inverse_transform`` are pure matmuls
against a fixed shared basis (no sign ambiguity).
"""
import numpy as np
import pytest

from nperf.cases import pca_fit, pca_inverse_transform, pca_transform
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# small CPU points (one per case): keep d below the GPU-eigh boundary -- this
# suite is CPU-only (conftest), the boundary matters only for the perf run.
_CASES = [
    (pca_fit, {'n': 512, 'd': 64, 'k': 16, 'seed': 0}, 'sklearn.PCA'),
    (pca_transform, {'n': 1024, 'd': 64, 'k': 16, 'seed': 0}, 'numpy.matmul'),
    (pca_inverse_transform,
     {'n': 1024, 'd': 64, 'k': 16, 'seed': 0}, 'numpy.matmul'),
]


@pytest.mark.parametrize('mod,p,refname', _CASES)
def test_baseline_shape(mod, p, refname):
    built = mod._build(p)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    # the only extra is the cupy GPU on-target twin.
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'


@pytest.mark.parametrize('mod,p,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, p, refname):
    built = mod._build(p)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        args = built.inputs_for(framework_of(provider_id))
        fid = compare(np.asarray(fn(*args)), built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_explained_variance_descending_and_positive():
    '''pca_fit's spectrum is positive and sorted descending (the top-k
    variances of a planted rank-k-dominant cov -- all above the floor).'''
    built = pca_fit._build({'n': 512, 'd': 64, 'k': 16, 'seed': 0})
    ev = np.asarray(built.fp64_reference)
    assert ev.shape == (16,)
    assert np.all(ev > 0)
    assert np.all(np.diff(ev) <= 1e-9)  # non-increasing


def test_transform_inverse_round_trip():
    '''Projecting then reconstructing onto the *same* top-k basis is a near
    identity on the planted signal (the noise floor is the only residual): the
    transform and inverse oracles compose to ~X (sign/rotation-invariant, the
    subspace shared), pinning that the two matmul cases are true inverses.'''
    from nperf.cases._pca import (
        np_basis,
        np_inverse,
        np_transform,
        pca_input,
    )
    x = pca_input(1024, 64, 16, seed=1)
    comps, mean = np_basis(x, 16)
    x_hat = np_inverse(np_transform(x, comps, mean), comps, mean)
    # reconstruction error is bounded by the discarded (noise) variance, small
    # vs the O(1) signal -- a loose but meaningful round-trip check.
    rel = np.linalg.norm(x_hat - x) / np.linalg.norm(x)
    assert rel < 0.2, f'round-trip rel-error {rel:.3g} too large'


def test_op_qualnames_match_nitrix():
    assert pca_fit.CASE.op_qualname == 'nitrix.stats.pca_fit'
    assert pca_transform.CASE.op_qualname == 'nitrix.stats.pca_transform'
    assert (pca_inverse_transform.CASE.op_qualname
            == 'nitrix.stats.pca_inverse_transform')
