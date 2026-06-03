# -*- coding: utf-8 -*-
"""Tier-2 graph cases (laplacian / modularity_matrix).

CPU build + oracle agreement for the runnable host baselines (jax + the
recognised graph references: scipy.sparse.csgraph for the Laplacian, networkx
for the modularity matrix); the cupy GPU ref is skipped here (needs a device +
the refs env).
"""
import numpy as np

from nperf.cases import laplacian, modularity_matrix
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'n': 48, 'seed': 0}
_CASES = [
    (laplacian, 'scipy.csgraph.laplacian'),
    (modularity_matrix, 'networkx.modularity_matrix'),
]


def test_baselines():
    for mod, refname in _CASES:
        names = set(mod._build(_P).baselines)
        assert {'nitrix-jax', refname} <= names
        assert mod._build(_P).ratio_reference == 'nitrix-jax'
        gpu = [n for n in names if n.startswith('cupy.')]
        assert len(gpu) == 1
        assert requires_of(mod._build(_P).baselines[gpu[0]][0]) == 'gpu'


def test_host_baselines_match_oracle():
    for mod, _ in _CASES:
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
    assert laplacian.CASE.op_qualname == 'nitrix.graph.laplacian'
    assert (modularity_matrix.CASE.op_qualname
            == 'nitrix.graph.modularity_matrix')
