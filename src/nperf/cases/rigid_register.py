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
from nitrix.register import (
    IndexSpace,
    RegistrationSpec,
    WorldSpace,
    rigid_register,
)

from ._base import BuiltPoint, Case, SlowBaseline
from ._real_anatomy import real_warp_pair
from ._register import (
    ants_register,
    dipy_register,
    warp_pair,
    warp_pair_cross_grid,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    levels, iters = int(param['levels']), int(param['iters'])
    spec = RegistrationSpec(levels=levels, iterations=iters)
    seed = param.get('seed', 0)
    if param.get('data') == 'mni152':
        # REAL anatomy: the MNI152 T1 under a planted rigid warp (same grid).
        moving, fixed = real_warp_pair(int(param.get('resolution', 2)), seed)
        space = IndexSpace()
        ants_ref = ants_register('Rigid')
        dipy_ref = dipy_register('rigid', levels, iters)
    elif param.get('space') == 'world':
        # cross-grid: fixed and moving on DIFFERENT grids (shape + anisotropic
        # spacing), recovered in physical space via WorldSpace; the refs get
        # the matching spacing/affines (their native physical-space regime).
        shape = tuple(param['shape'])
        f_sp = tuple(param.get('fixed_spacing', (1.0, 1.0, 1.0)))
        m_sp = tuple(param.get('moving_spacing', (1.2, 1.0, 0.9)))
        moving, fixed, a_m, a_f = warp_pair_cross_grid(
            shape, tuple(param['moving_shape']),
            fixed_spacing=f_sp, moving_spacing=m_sp, seed=seed)
        space = WorldSpace(fixed_affine=jnp.asarray(a_f),
                           moving_affine=jnp.asarray(a_m))
        ants_ref = ants_register('Rigid', spacing=(f_sp, m_sp))
        dipy_ref = dipy_register('rigid', levels, iters, affines=(a_f, a_m))
    else:
        moving, fixed = warp_pair(tuple(param['shape']), seed)
        space = IndexSpace()  # the default (shared-grid) path, unchanged
        ants_ref = ants_register('Rigid')
        dipy_ref = dipy_register('rigid', levels, iters)
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
            'jax',
            lambda mv, fx: rigid_register(mv, fx, spec=spec,
                                          space=space).params),
        'ants.registration': ('ants', ants_ref),   # ANTs Rigid (physical sp.)
        'dipy.registration': ('dipy', dipy_ref),    # dipy rigid MI
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
# Cross-grid points (the realistic cross-resolution/cross-modal regime): fixed
# and moving on DIFFERENT grids + anisotropic moving spacing, recovered via
# WorldSpace (added alongside the shared-grid points -- the refs' native
# physical-space regime, so the fairest cross-tool comparison).
_LARGE_WORLD = [
    {'shape': [96, 96, 96], 'moving_shape': [80, 96, 112],
     'levels': 2, 'iters': 20, 'seed': 0, 'space': 'world',
     'fixed_spacing': [1, 1, 1], 'moving_spacing': [1.2, 1.0, 0.9]},
    {'shape': [128, 128, 128], 'moving_shape': [112, 128, 144],
     'levels': 2, 'iters': 20, 'seed': 0, 'space': 'world',
     'fixed_spacing': [1, 1, 1], 'moving_spacing': [1.2, 1.0, 0.9]},
]
# Real-anatomy point: the MNI152 T1 (~99x117x95 @2mm) under a planted rigid
# warp -- real edges/intensity (realistic difficulty), exact ground-truth warp.
_LARGE_REAL = [{'data': 'mni152', 'resolution': 2, 'levels': 2, 'iters': 20,
                'seed': 0}]

CASE = Case(
    name='rigid_register',
    op_qualname='nitrix.register.rigid_register',
    tier='marquee',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 10, 'seed': 0},
    large_param_points=tuple(
        [{'shape': s, 'levels': 2, 'iters': 20, 'seed': 0} for s in _LARGE]
        + _LARGE_WORLD + _LARGE_REAL),
    # dipy MI is slow at scale (128^3 ~35 s on CPU); skippable for dev cycles.
    slow_baselines=(SlowBaseline(
        'dipy.registration',
        reason='dipy MI ~35 s at 128^3 on CPU (CPU-only cython); '
               'worker-timeout-capped in the full matrix.'),),
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
