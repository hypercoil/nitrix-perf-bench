# -*- coding: utf-8 -*-
"""Tier-2 geometry registration / deformation-field cases.

CPU build + oracle agreement for the runnable host baselines (jax + the numpy /
scipy.ndimage reimplementations of the exact nitrix conventions); the cupy GPU
refs are skipped here (need a device + the refs env). The numpy floors are the
warranted-claim check: nitrix's central-diff Jacobian and scaling-and-squaring
flow match the reimplemented conventions (not numpy.gradient at the boundary --
see _registration.py).
"""
import numpy as np

from nperf.cases import (
    integrate_velocity_field,
    jacobian_det_displacement,
    jacobian_displacement,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# (module, small param, expected reference baseline name)
_CASES = [
    (jacobian_displacement, {'d': 12, 'seed': 0}, 'numpy.jacobian'),
    (jacobian_det_displacement, {'d': 12, 'seed': 0}, 'numpy.jacobian_det'),
    (integrate_velocity_field, {'d': 10, 'seed': 0},
     'scipy.ndimage.map_coordinates'),
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
    assert (jacobian_displacement.CASE.op_qualname
            == 'nitrix.geometry.jacobian_displacement')
    assert (jacobian_det_displacement.CASE.op_qualname
            == 'nitrix.geometry.jacobian_det_displacement')
    assert (integrate_velocity_field.CASE.op_qualname
            == 'nitrix.geometry.integrate_velocity_field')
