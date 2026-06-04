# -*- coding: utf-8 -*-
"""Tier-2 geometry equirectangular sphere-grid pad / unpad cases.

CPU build + oracle agreement for the runnable host baselines (jax + the numpy
reimplementation of the exact wrap / pole-flip / slice); the cupy GPU refs are
skipped here (need a device + the refs env). The pad topology is
nitrix-specific (no external library), so the numpy floor is the
warranted-claim check, plus the exact unpad(pad(x)) == x round-trip.
"""
import numpy as np

from nperf.cases import sphere_grid_pad_2d, sphere_grid_unpad_2d
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_CASES = [
    (sphere_grid_pad_2d, 'numpy.sphere_grid_pad'),
    (sphere_grid_unpad_2d, 'numpy.sphere_grid_unpad'),
]
_P = {'h': 16, 'seed': 0}


def test_baselines():
    for mod, refname in _CASES:
        names = set(mod._build(_P).baselines)
        assert {'nitrix-jax', refname} <= names, mod.CASE.name
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


def test_pad_unpad_roundtrip():
    '''sphere_grid_unpad_2d strips exactly what sphere_grid_pad_2d added.'''
    from nperf.cases._sphere_grid import (
        np_sphere_pad,
        np_sphere_unpad,
        sphere_grid_input,
    )
    img = sphere_grid_input(16)
    pad = sphere_grid_pad_2d._PAD
    back = np_sphere_unpad(pad)(np_sphere_pad(pad)(img))
    assert back.shape == img.shape
    assert np.abs(back - img).max() == 0.0


def test_op_qualnames():
    assert (sphere_grid_pad_2d.CASE.op_qualname
            == 'nitrix.geometry.sphere_grid_pad_2d')
    assert (sphere_grid_unpad_2d.CASE.op_qualname
            == 'nitrix.geometry.sphere_grid_unpad_2d')
