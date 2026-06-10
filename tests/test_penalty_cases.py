# -*- coding: utf-8 -*-
"""Registration penalty cases: gradient_smoothness / bending_energy /
jacobian_folding_penalty.

Pins the **warranted comparison** (the numpy reimplementation matches nitrix's
exact central-diff / 'nearest'-boundary convention -- not numpy.gradient) and
the case contract + the brain-scale size tier. The folding case is built at a
deliberately-folding input scale so its ``relu(-det)`` branch is exercised (a
realistic small warp folds nowhere -> a degenerate 0).
"""
import numpy as np
import pytest

from nperf.cases import bending_energy as bending_mod
from nperf.cases import gradient_smoothness as grad_mod
from nperf.cases import jacobian_folding_penalty as fold_mod
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_MODS = [grad_mod, bending_mod, fold_mod]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_baselines_and_contract(mod):
    built = mod._build(mod.CASE.representative)
    name = mod.CASE.name
    assert set(built.baselines) == {
        'nitrix-jax', f'numpy.{name}', f'cupy.{name}'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is not None  # a real fp64 oracle
    assert mod.CASE.op_qualname == f'nitrix.register.{name}'
    # ships with a brain-scale size tier (COVERAGE_MANDATE §7-D).
    assert mod.CASE.large_param_points
    big = max(p['d'] for p in mod.CASE.large_param_points)
    assert big > mod.CASE.representative['d']
    assert mod.CASE.complexity and 'O(N)' in mod.CASE.complexity


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_host_baselines_match_oracle(mod):
    # nitrix + the numpy reimpl must both match the fp64 oracle at every dev
    # point -- the warranted-convention pin (cupy needs a device, skipped).
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


def test_folding_input_actually_folds():
    # The folding case must exercise the relu(-det) branch -- a non-degenerate
    # (nonzero) penalty -- else the fidelity check is the trivial 0 == 0.
    built = fold_mod._build(fold_mod.CASE.representative)
    assert float(np.asarray(built.fp64_reference)) > 1e-4
