# -*- coding: utf-8 -*-
"""Morphology pooling cases: max_pool_with_indices_nd / max_unpool_nd.

Pins the warranted comparison (pool values == numpy windowed-max; unpool ==
numpy scatter) + the case contracts + the pool->unpool round-trip (the index
convention: unpool places each window's max back at its argmax position).
"""
import numpy as np
import pytest

from nperf.cases import max_pool_with_indices_nd as pool_mod
from nperf.cases import max_unpool_nd as unpool_mod
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_MODS = [pool_mod, unpool_mod]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_contract_and_oracle(mod):
    case = mod.CASE
    built = mod._build(case.representative)
    name = case.name
    short = 'max_pool' if name == 'max_pool_with_indices_nd' else 'max_unpool'
    assert set(built.baselines) == {
        'nitrix-jax', f'numpy.{short}', f'cupy.{short}'}
    assert built.ratio_reference == 'nitrix-jax'
    assert case.op_qualname == f'nitrix.morphology.{name}'
    assert case.large_param_points
    # warranted: nitrix + numpy match the fp64 oracle exactly at every point.
    for p in case.param_points:
        b = mod._build(p)
        for bname, (pid, fn) in b.baselines.items():
            if pid == 'cupy':
                continue
            out = np.asarray(fn(*b.inputs_for(framework_of(pid))), np.float64)
            fid = compare(out, b.fp64_reference, rtol=case.rtol,
                          atol=case.atol)
            assert fid['status'] == 'pass', f'{name}/{bname}@{p}'


def test_pool_unpool_round_trip_recovers_maxima():
    # unpool(pool(x).values, pool(x).indices) places each window's max back at
    # its argmax voxel (0 elsewhere): the nonzeros are exactly the maxima
    # and they equal x there -- pins the index (global-flat) convention.
    import jax.numpy as jnp
    from nitrix.morphology import max_pool_with_indices_nd, max_unpool_nd

    from nperf.cases._pooling import pool_input
    x = pool_input(16, seed=2)
    xj = jnp.asarray(x)
    vals, idx = max_pool_with_indices_nd(xj, pool_size=2, spatial_rank=3)
    un = np.asarray(max_unpool_nd(vals, idx, output_shape=[16, 16, 16],
                                  spatial_rank=3))
    vals = np.asarray(vals)
    assert int((un != 0).sum()) == vals.size  # one max per window
    assert np.allclose(np.sort(un[un != 0]), np.sort(vals.ravel()), atol=1e-6)
    # every nonzero equals x at that voxel (the recovered maximum)
    assert np.all((un == 0) | (un == x))
