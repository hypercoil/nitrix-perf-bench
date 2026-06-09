# -*- coding: utf-8 -*-
"""Registration recipe case (R1): nitrix.register.rigid_register.

A task-level end-to-end driver: no shared cross-impl oracle (fp64_reference is
None), so this pins (a) the case contract and (b) that the recipe actually
recovers the planted warp -- the accuracy pin the no-oracle bench can't gate.
The recovery test compiles the (unrolled) recipe once, so it is the slow test
in this file by design.
"""
import jax.numpy as jnp
import numpy as np
from nitrix.register import RegistrationSpec, rigid_register

from nperf.cases import rigid_register as case_mod
from nperf.cases._register import ncc, warp_pair


def test_case_contract():
    built = case_mod._build(case_mod.CASE.representative)
    assert set(built.baselines) == {'nitrix-jax', 'ants.registration'}
    assert built.ratio_reference == 'nitrix-jax'
    # task-level: no shared oracle, but a documented reason + the compile law.
    assert built.fp64_reference is None and built.fidelity_note
    assert case_mod.CASE.complexity and 'compile' in case_mod.CASE.complexity
    # param points span the unrolled iteration count (the compile axis).
    iters = {(p['levels'], p['iters']) for p in case_mod.CASE.param_points}
    assert len(iters) >= 2


def test_op_qualname():
    assert case_mod.CASE.op_qualname == 'nitrix.register.rigid_register'


def test_recipe_recovers_planted_warp():
    # The accuracy pin: registering moving onto fixed must improve the
    # alignment (the benched config exercises a *working* registration, not a
    # degenerate fast-but-wrong one). Coarse-to-fine so few iters suffice.
    moving, fixed = warp_pair([28, 28, 28], seed=0)
    res = rigid_register(jnp.asarray(moving), jnp.asarray(fixed),
                         spec=RegistrationSpec(levels=2, iterations=15))
    warped = np.asarray(res.warped)
    before = ncc(moving, fixed)
    after = ncc(warped, fixed)
    assert after > before + 0.05, f'no improvement {before:.3f}->{after:.3f}'
    assert np.isfinite(np.asarray(res.params)).all()
