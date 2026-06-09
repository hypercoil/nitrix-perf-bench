# -*- coding: utf-8 -*-
"""Registration recipe cases (R1-R2): rigid / affine / diffeomorphic-demons.

Task-level end-to-end drivers: no shared cross-impl oracle (fp64_reference is
None), so these pin (a) the case contract and (b) that each recipe actually
recovers the planted warp -- the accuracy pin the no-oracle bench can't gate.
The recovery tests compile the (unrolled) recipe once each, so they are the
slow tests in this file by design.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.register import (
    DemonsSpec,
    RegistrationSpec,
    affine_register,
    diffeomorphic_demons_register,
    rigid_register,
)

from nperf.cases import affine_register as affine_mod
from nperf.cases import diffeomorphic_demons as demons_mod
from nperf.cases import rigid_register as rigid_mod
from nperf.cases._register import ncc, warp_pair

_MODS = [rigid_mod, affine_mod, demons_mod]
# (recipe fn, spec) per case -- coarse-to-fine, few iters (recovery test).
_RECOVER = [
    (rigid_register, RegistrationSpec(levels=2, iterations=15)),
    (affine_register, RegistrationSpec(levels=2, iterations=15)),
    (diffeomorphic_demons_register, DemonsSpec(levels=2, iterations=20)),
]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_case_contract(mod):
    built = mod._build(mod.CASE.representative)
    assert set(built.baselines) == {'nitrix-jax', 'ants.registration',
                                     'dipy.registration'}
    assert built.ratio_reference == 'nitrix-jax'
    # task-level: no shared oracle, but a documented reason + the compile law.
    assert built.fp64_reference is None and built.fidelity_note
    assert mod.CASE.complexity and 'compile' in mod.CASE.complexity
    assert mod.CASE.op_qualname.startswith('nitrix.register.')
    # param points span the unrolled iteration count (the compile axis).
    iters = {(p['levels'], p['iters']) for p in mod.CASE.param_points}
    assert len(iters) >= 2


@pytest.mark.parametrize('recipe,spec', _RECOVER,
                         ids=['rigid', 'affine', 'demons'])
def test_recipe_recovers_planted_warp(recipe, spec):
    # The accuracy pin: registering moving onto fixed must improve the
    # alignment (each benched recipe does *working* registration, not a
    # degenerate fast-but-wrong one).
    moving, fixed = warp_pair([28, 28, 28], seed=0)
    res = recipe(jnp.asarray(moving), jnp.asarray(fixed), spec=spec)
    before = ncc(moving, fixed)
    after = ncc(np.asarray(res.warped), fixed)
    assert after > before + 0.05, f'no improvement {before:.3f}->{after:.3f}'
