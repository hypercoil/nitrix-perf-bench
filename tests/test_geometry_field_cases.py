# -*- coding: utf-8 -*-
"""Geometry field-algebra cases: spatial_gradient / invert_displacement /
compose_velocity.

Pins the warranted comparison (numpy reimpls match nitrix's exact central-diff
/ fixed-point conventions), the case contract, the size tier, and -- for
invert_displacement -- that the fixed point actually inverts the warp.
"""
import numpy as np
import pytest

from nperf.cases import compose_velocity as compose_mod
from nperf.cases import invert_displacement as invert_mod
from nperf.cases import spatial_gradient as grad_mod
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_MODS = [grad_mod, invert_mod, compose_mod]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_baselines_and_contract(mod):
    built = mod._build(mod.CASE.representative)
    name = mod.CASE.name
    assert set(built.baselines) == {
        'nitrix-jax', f'numpy.{name}', f'cupy.{name}'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is not None
    assert mod.CASE.op_qualname == f'nitrix.geometry.{name}'
    assert mod.CASE.large_param_points
    big = max(p['d'] for p in mod.CASE.large_param_points)
    assert big > mod.CASE.representative['d']
    assert mod.CASE.complexity


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_host_baselines_match_oracle(mod):
    case = mod.CASE
    for p in case.param_points:
        built = mod._build(p)
        for bname, (pid, fn) in built.baselines.items():
            if pid == 'cupy':
                continue
            out = np.asarray(fn(*built.inputs_for(framework_of(pid))),
                             np.float64)
            fid = compare(out, built.fp64_reference,
                          rtol=case.rtol, atol=case.atol)
            assert fid['status'] == 'pass', (
                f'{case.name}/{bname}@{p}: rel_to_tol={fid["rel_to_tol"]:.3g}')


def test_invert_displacement_actually_inverts():
    # s_inv + s∘(id + s_inv) ≈ 0 : the fixed point inverts the warp (so the
    # case measures a *correct* inverse, not a half-converged one).
    import scipy.ndimage as ndi

    from nperf.cases._registration import displacement_input
    s_inv = np.asarray(invert_mod._build({'d': 24, 'seed': 0}).fp64_reference)
    s = displacement_input((24, 24, 24), 3, 0, scale=0.1).astype(np.float64)
    sp = s.shape[:-1]
    idg = np.stack(np.meshgrid(*[np.arange(n) for n in sp], indexing='ij'),
                   -1).astype(np.float64)
    coords = np.moveaxis(idg + s_inv, -1, 0).reshape(3, -1)
    warped = np.stack([ndi.map_coordinates(s[..., k], coords, order=1,
                                           mode='nearest').reshape(sp)
                       for k in range(3)], -1)
    assert np.abs(s_inv + warped).max() < 1e-3
