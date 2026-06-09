# -*- coding: utf-8 -*-
"""Registration recipe (R1): ``nitrix.register.rigid_register`` -- the driver.

An **end-to-end** rigid registration (coarse-to-fine Gauss-Newton/LM on SSD).
Task-level: nitrix / ANTs / dipy converge to different transforms, so there is
**no shared oracle** (``fp64_reference`` is ``None``); recovery accuracy is
pinned in ``tests/test_register_cases.py``.

**Post loop-roll (nitrix ``ddc2e10``): steady_time is the headline, not
compile.**  The optimiser loop is now a ``lax.scan`` (was Python-unrolled), so
the cold compile is **~flat in iterations** and small: on the L4, ``L1x10`` ->
``L3x30`` compiles ~4.3 s -> ~9.1 s, and that ~2x is the *levels* (3 pyramid
graphs), not the iterations -- demons ``L2x20`` and ``L2x40`` compile
*identically*.  This **closes the "registration slow on GPU" diagnosis** (filed
``registration-recipe-cold-compile``, now resolved): the old default ``L3x30``
went 145 s -> ~9 s compile, and the affine sibling's ``L3x30`` *CPU* compile,
which used to fail XLA outright, now compiles.  The suite still splits compile
from steady; the dev configs vary ``(levels, iterations)`` to *show* compile is
now flat in iters, and the size tier (``large_param_points``) varies the volume
-- the axis steady actually scales on (see ``complexity``).

Two task-level domain references (own refs env, CPU, not jit-compiled, so
wall-clock is the full registration with no separate compile) -- read against
nitrix's steady + one-time compile: ANTsPy ``registration(Rigid)`` (fixed
internal schedule) and **dipy** rigid (MI, pyramid driven by this case's
``levels`` x ``iters`` -- see ``cases/_register.py``). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import RegistrationSpec, rigid_register

from ._base import BuiltPoint, Case
from ._register import ants_register, dipy_register, warp_pair


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    spec = RegistrationSpec(levels=int(param['levels']),
                            iterations=int(param['iters']))
    moving, fixed = warp_pair(shape, param.get('seed', 0))
    mj = jax.block_until_ready(jnp.asarray(moving))
    fj = jax.block_until_ready(jnp.asarray(fixed))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (mj, fj)
        return (moving, fixed)  # numpy arrays for ANTs / dipy

    baselines = {
        # the recipe; return params (the deliverable -- forces the full scan to
        # run, so compile_time is the real cold compile).
        'nitrix-jax': (
            'jax', lambda mv, fx: rigid_register(mv, fx, spec=spec).params),
        'ants.registration': (  # task-level domain ref (ANTs Rigid)
            'ants', ants_register('Rigid')),
        'dipy.registration': (  # task-level domain ref (dipy rigid MI)
            'dipy', dipy_register('rigid', int(param['levels']),
                                  int(param['iters']))),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level recipe: no shared oracle (nitrix GN/LM, '
                       'ANTs, dipy converge to different transforms). Reads '
                       'steady_time (the headline post loop-roll) + the '
                       'one-time compile; recovery pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: spec configs at one small volume, varying (levels, iterations) to
# *show* the compile is now flat in iters (post lax.scan). Representative =
# the fast L1x10 (cheap, for --quick / drift).
_CONFIGS = [(1, 10), (2, 20), (3, 30)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): fix a mid config and vary the volume -- the axis the
# steady cost scales on (~ N voxels). rigid is the lightest recipe (P=6, no SVF
# field), so it carries the most HBM headroom -> runs to 192^3.
_LARGE = [[96, 96, 96], [128, 128, 128], [160, 160, 160], [192, 192, 192]]

CASE = Case(
    name='rigid_register',
    op_qualname='nitrix.register.rigid_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 10, 'seed': 0},
    large_param_points=tuple(
        {'shape': s, 'levels': 2, 'iters': 20, 'seed': 0} for s in _LARGE),
    complexity=(
        'post loop-roll (lax.scan): COMPILE ~flat in iterations AND volume '
        '(XLA compiles the per-iteration op graph, not an unrolled chain) -- '
        '~4-11 s across configs/sizes (was 16-211 s unrolled). STEADY is the '
        'headline ~ iterations x P x N: each LM iter assembles the small-P '
        'normal equations J^TJ (P=6; ~P forward warp-passes + a P x P solve). '
        'GPU steady is overhead-bound (~flat) below ~48^3 then compute-bound '
        '(~N); the GPU/CPU speedup climbs from ~4x (24^3) to a brain-scale '
        'plateau ~25x. HBM: lighter than demons, but cold peak_hbm is '
        'autotune-contaminated at large N -- no OOM projection (see '
        'reports/REGISTRATION_SCALING.md). Bias: the size tier fixes '
        '(levels=2, iters=20); real pipelines raise levels with resolution.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
