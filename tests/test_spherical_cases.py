# -*- coding: utf-8 -*-
"""Tier-2 geometry spherical distance + compactness cases.

CPU build + oracle agreement for the runnable host baselines (jax + the
references: sklearn.haversine_distances for the geodesic -- the canonical
domain-tool great-circle distance -- and the numpy reimplementation for the
compactness penalty); the cupy GPU refs are skipped here (need a device + the
refs env). The geodesic row is also the warranted-claim check that nitrix's
atan2(|X×Y|, X·Y) formula equals the sklearn haversine angular distance.
"""
import numpy as np

from nperf.cases import compactness_penalty, spherical_geodesic_distance
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# (module, small param, expected reference baseline name)
_CASES = [
    (spherical_geodesic_distance, {'n': 64, 'seed': 0}, 'sklearn.haversine'),
    (compactness_penalty, {'p': 256, 'seed': 0}, 'numpy.compactness'),
]


def test_baselines():
    for mod, p, refname in _CASES:
        names = set(mod._build(p).baselines)
        assert {'nitrix-jax', refname} <= names, mod.CASE.name
        assert mod._build(p).ratio_reference == 'nitrix-jax'
        gpu = [n for n in names if n.startswith('cupy.')]
        assert len(gpu) == 1
        assert requires_of(mod._build(p).baselines[gpu[0]][0]) == 'gpu'


def test_host_baselines_match_oracle():
    for mod, p, _ in _CASES:
        built = mod._build(p)
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
    assert (spherical_geodesic_distance.CASE.op_qualname
            == 'nitrix.geometry.spherical_geodesic_distance')
    assert (compactness_penalty.CASE.op_qualname
            == 'nitrix.geometry.compactness_penalty')
