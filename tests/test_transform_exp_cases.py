# -*- coding: utf-8 -*-
"""Lie transform-exp cases: rigid_exp / affine_exp / rigid_log (batched).

Pins the warranted comparison (the vectorised numpy reimplementations match
nitrix's exact Rodrigues / matrix_exp / SO(3)-log conventions), the batched
exp/log round-trip, the case contract, and the batch size tier.
"""
import numpy as np
import pytest

from nperf.cases import affine_exp as affine_mod
from nperf.cases import rigid_exp as rigid_mod
from nperf.cases import rigid_log as log_mod
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_MODS = [rigid_mod, affine_mod, log_mod]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_baselines_and_contract(mod):
    built = mod._build(mod.CASE.representative)
    name = mod.CASE.name
    assert set(built.baselines) == {
        'nitrix-jax', f'numpy.{name}', f'cupy.{name}'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is not None
    assert mod.CASE.op_qualname == f'nitrix.geometry.{name}'
    assert mod.CASE.output_independent  # per-transform independent (batched)
    assert mod.CASE.large_param_points
    assert max(p['b'] for p in mod.CASE.large_param_points) > (
        mod.CASE.representative['b'])
    assert mod.CASE.complexity and 'O(B)' in mod.CASE.complexity


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


def test_exp_log_round_trip():
    # rigid_log o rigid_exp == identity on the parameters (the convention pin).
    import jax.numpy as jnp
    from nitrix.geometry import rigid_exp, rigid_log

    from nperf.cases._transforms import rigid_params
    p = rigid_params(512, seed=3)
    rec = np.asarray(rigid_log(rigid_exp(jnp.asarray(p), ndim=3), ndim=3))
    assert np.abs(rec - p).max() < 1e-4
