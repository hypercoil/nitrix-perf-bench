# -*- coding: utf-8 -*-
"""Tier-2 geometry coordinate-conversion cases.

CPU build + oracle agreement for the runnable host baselines (jax + the numpy
closed-form references); the cupy GPU refs are skipped here (need a device +
the refs env). The coordinate transforms are pure trig whose closed form is the
reference, so this also checks the latlong<->cartesian round-trip (both
directions match the same fp64 closed form).
"""
import numpy as np

from nperf.cases import cartesian_to_latlong, latlong_to_cartesian
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# (module, small param, expected reference baseline name)
_CASES = [
    (latlong_to_cartesian, {'n': 256, 'seed': 0},
     'numpy.latlong_to_cartesian'),
    (cartesian_to_latlong, {'n': 256, 'seed': 0},
     'numpy.cartesian_to_latlong'),
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


def test_roundtrip_latlong():
    '''latlong -> cartesian -> latlong recovers the input angles (the
    closed-form inverse check, fp64).'''
    from nperf.cases._sphere import (
        np_cartesian_to_latlong,
        np_latlong_to_cartesian,
    )
    ll = np.stack([
        np.linspace(-1.5, 1.5, 64),
        np.linspace(-3.0, 3.0, 64),
    ], axis=-1)
    back = np_cartesian_to_latlong(np_latlong_to_cartesian(ll))
    assert np.abs(back - ll).max() < 1e-12


def test_op_qualnames():
    assert (latlong_to_cartesian.CASE.op_qualname
            == 'nitrix.geometry.latlong_to_cartesian')
    assert (cartesian_to_latlong.CASE.op_qualname
            == 'nitrix.geometry.cartesian_to_latlong')
