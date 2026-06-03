# -*- coding: utf-8 -*-
"""Tier-2 spectral-embedding case (laplacian_eigenmap).

The eigenvalues (sign-unambiguous) are the fidelity target: both nitrix solver
paths -- lobpcg (matrix-free, the genuine-GPU path) and eigh (safe_eigh) -- and
scipy.sparse.eigsh must all match the fp64 normalised-Laplacian eigenvalue
oracle. The cupy GPU ref is skipped here (needs a device + the refs env).
"""
import numpy as np

from nperf.cases import diffusion_embedding as de
from nperf.cases import laplacian_eigenmap as le
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'n': 128, 'seed': 0}
_CASES = [le, de]  # the spectral pair: both share the lobpcg/eigh solver paths


def test_baselines_two_solvers():
    for mod in _CASES:
        built = mod._build(_P)
        assert set(built.baselines) == {
            'nitrix-jax', 'nitrix-jax-eigh',
            'scipy.sparse.eigsh', 'cupyx.sparse.eigsh'}
        assert built.ratio_reference == 'nitrix-jax'  # the lobpcg path
        assert requires_of(built.baselines['cupyx.sparse.eigsh'][0]) == 'gpu'


def test_host_eigenvalues_match_oracle():
    '''lobpcg + eigh + scipy.eigsh recover the nontrivial spectral eigenvalues
    (smallest for the Laplacian, largest for the diffusion operator; lobpcg to
    its iterative tolerance).'''
    for mod in _CASES:
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
    assert le.CASE.op_qualname == 'nitrix.graph.laplacian_eigenmap'
    assert de.CASE.op_qualname == 'nitrix.graph.diffusion_embedding'
