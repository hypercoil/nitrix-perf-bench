# -*- coding: utf-8 -*-
"""Tier-2 graph cases (the adjacency + community family).

CPU build + oracle agreement for the runnable host baselines (jax + the
recognised graph references: scipy.sparse.csgraph for the Laplacian, networkx
for the modularity matrix + the Newman quality score); the cupy GPU refs are
skipped here (need a device + the refs env). For ``relaxed_modularity`` the
networkx baseline is the canonical Newman modularity / 2 -- so this test is
also the warranted-claim check that nitrix's score (hard one-hot partition,
``exclude_diag=False``) equals that bridged reference.
"""
import numpy as np

from nperf.cases import (
    coaffiliation,
    degree_vector,
    girvan_newman_null,
    laplacian,
    modularity_matrix,
    relaxed_modularity,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'n': 48, 'seed': 0}
_CASES = [
    (laplacian, 'scipy.csgraph.laplacian'),
    (modularity_matrix, 'networkx.modularity_matrix'),
    (degree_vector, 'numpy.degree'),
    (girvan_newman_null, 'numpy.gn_null'),
    (coaffiliation, 'numpy.coaffiliation'),
    (relaxed_modularity, 'networkx.modularity'),
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
    assert degree_vector.CASE.op_qualname == 'nitrix.graph.degree_vector'
    assert (girvan_newman_null.CASE.op_qualname
            == 'nitrix.graph.girvan_newman_null')
    assert coaffiliation.CASE.op_qualname == 'nitrix.graph.coaffiliation'
    assert (relaxed_modularity.CASE.op_qualname
            == 'nitrix.graph.relaxed_modularity')
