# -*- coding: utf-8 -*-
"""Tier-2 geometry centre-of-mass cluster.

CPU build + oracle agreement for the runnable host baselines (jax + the
references: scipy.ndimage.center_of_mass for the grid centroid -- the canonical
imaging reference -- and the numpy weighted mean for the point-cloud form); the
cupy GPU refs are skipped here (need a device + the refs env). For the grid
cases the scipy.ndimage baseline is also the warranted-claim check that
nitrix's centroid equals that canonical reference.
"""
import numpy as np

from nperf.cases import (
    center_of_mass_grid,
    center_of_mass_points,
    displacement_from_reference_grid,
    displacement_from_reference_points,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# (module, small param, expected reference baseline name)
_CASES = [
    (center_of_mass_grid, {'d': 12, 'seed': 0},
     'scipy.ndimage.center_of_mass'),
    (center_of_mass_points, {'p': 256, 'seed': 0}, 'numpy.weighted_mean'),
    (displacement_from_reference_grid, {'d': 12, 'seed': 0},
     'scipy.ndimage.center_of_mass'),
    (displacement_from_reference_points, {'p': 256, 'seed': 0},
     'numpy.weighted_mean'),
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
    assert (center_of_mass_grid.CASE.op_qualname
            == 'nitrix.geometry.center_of_mass_grid')
    assert (center_of_mass_points.CASE.op_qualname
            == 'nitrix.geometry.center_of_mass_points')
    assert (displacement_from_reference_grid.CASE.op_qualname
            == 'nitrix.geometry.displacement_from_reference_grid')
    assert (displacement_from_reference_points.CASE.op_qualname
            == 'nitrix.geometry.displacement_from_reference_points')
