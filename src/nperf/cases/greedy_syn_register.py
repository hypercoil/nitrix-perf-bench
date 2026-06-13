# -*- coding: utf-8 -*-
"""Registration recipe (R2 v2): ``nitrix.register.greedy_syn_register``.

Greedy **symmetric diffeomorphic** registration (SyN-style): symmetric
forward/inverse velocity fields driven to a midpoint by a local-NCC force, the
canonical deformable recipe.  Task-level (no shared oracle: nitrix LNCC, ANTs'
CC, and dipy's SSD converge to *different* warps), so ``fp64_reference`` is
``None`` and the diffeomorphic recovery (``ncc`` up, ``jacobian_det > 0``) is
pinned in ``tests/test_register_cases.py``.

The headline read is nitrix **steady** (warm) + its **one-time compile** vs the
gold standard's full wall-clock.  Two CPU domain references (own refs envs, not
jit-compiled): **ANTsPy** ``registration(SyNOnly)`` -- the canonical SyN, run
**deformable-only** to match nitrix's greedy SyN (no affine pre-step) -- and
**dipy** ``SymmetricDiffeomorphic`` (SSD).  Crucially, ANTs SyNOnly is **fast**
on CPU (measured here ~0.47 / 2.9 / 6.0 s at 48 / 96 / 128^3 -- ITK-backed
C++), so it is **not** a slow_baseline; **dipy** is the genuinely slow one
(~126 s at 128^3, CPU-only cython).  Because even this deformable gold standard
is only *seconds* on CPU, the GPU win must be **earned** -- it has to clear the
~4x hardware-cost bar to count, which the economic verdict measures rather than
assumes (see ``tools/economic_report.py``).  Ratio vs ``nitrix-jax``.

The size tier also carries **anisotropic** points (``spacing=[1,1,3]``): the op
corrects the bias where a voxel-isotropic Gaussian / force is physically
anisotropic (``SyNSpec.spacing``); the refs get the matching voxel->world
affine so the comparison stays in physical space.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import SyNSpec, greedy_syn_register

from ._base import BuiltPoint, Case, SlowBaseline
from ._real_anatomy import real_syn_pair
from ._register import (
    _affine,
    aniso_pair,
    ants_register,
    dipy_register,
    syn_pair,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    levels, iters = int(param['levels']), int(param['iters'])
    spacing = param.get('spacing')  # None -> isotropic (voxel space)
    seed = param.get('seed', 0)
    if param.get('data') == 'mni152':
        # REAL anatomy: MNI152 T1 under a smooth non-rigid warp (a small
        # background noise floor; see _real_anatomy).
        moving, fixed = real_syn_pair(int(param.get('resolution', 2)), seed)
        spec = SyNSpec(levels=levels, iterations=iters)
        ants_ref = ants_register('SyNOnly')
        dipy_ref = dipy_register('syn', levels, iters)
    elif spacing is not None:
        moving, fixed, sp = aniso_pair(tuple(param['shape']), spacing, seed)
        spec = SyNSpec(levels=levels, iterations=iters, spacing=sp)
        aff = _affine(spacing)
        ants_ref = ants_register('SyNOnly', spacing=list(spacing))
        dipy_ref = dipy_register('syn', levels, iters, affines=(aff, aff))
    else:
        moving, fixed = syn_pair(tuple(param['shape']), seed)
        spec = SyNSpec(levels=levels, iterations=iters)
        ants_ref = ants_register('SyNOnly')
        dipy_ref = dipy_register('syn', levels, iters)

    mj = jax.block_until_ready(jnp.asarray(moving))
    fj = jax.block_until_ready(jnp.asarray(fixed))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (mj, fj)
        return (moving, fixed)  # numpy for ANTs / dipy

    baselines = {
        # return the displacement field (the deliverable): forces the full
        # forward+inverse integrate + midpoint compose to run (real compile).
        'nitrix-jax': (
            'jax', lambda mv, fx: greedy_syn_register(mv, fx, spec=spec
                                                      ).displacement),
        'ants.registration': ('ants', ants_ref),  # gold std (fast: SyNOnly)
        'dipy.registration': ('dipy', dipy_ref),   # dipy diffeo (the slow one)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level deformable recipe: no shared oracle '
                       '(nitrix LNCC vs ANTs CC vs dipy SSD converge to '
                       'different warps). Reads steady + one-time compile; '
                       'recovery (ncc up, jacobian_det>0) pinned in tests.'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: configs at one small volume, varying (levels, iterations) to show
# the per-level lax.scan keeps compile ~flat in iters. Representative = the
# cheap L1x40 (for --quick / drift).
_CONFIGS = [(1, 40), (2, 80), (3, 80)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): vary the volume at a fixed mid config (SyN is
# identical-shape / intra-pair, so only the spatial axis scales), plus
# anisotropic points (1x1x3) at two sizes.
_LARGE = [{'shape': s, 'levels': 2, 'iters': 80, 'seed': 0}
          for s in ([64, 64, 64], [96, 96, 96], [128, 128, 128])]
_LARGE += [{'shape': s, 'levels': 2, 'iters': 80, 'seed': 0,
            'spacing': [1, 1, 3]} for s in ([64, 64, 64], [96, 96, 96])]
# Real-anatomy point: MNI152 T1 (~99^3 @2mm) under a smooth non-rigid warp.
_LARGE += [{'data': 'mni152', 'resolution': 2, 'levels': 2, 'iters': 80,
            'seed': 0}]

CASE = Case(
    name='greedy_syn_register',
    op_qualname='nitrix.register.greedy_syn_register',
    tier='marquee',
    output_independent=False,  # a global iterative deformable fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 40, 'seed': 0},
    large_param_points=tuple(_LARGE),
    # Only dipy is slow: ANTs SyNOnly is FAST (~6 s at 128^3, measured), so it
    # stays in dev cycles as the gold-standard bar; dipy SyN is the slow one.
    slow_baselines=(
        SlowBaseline('dipy.registration',
                     reason='dipy SymmetricDiffeomorphic ~126 s at 128^3 on '
                            'CPU (CPU-only cython, super-linear); '
                            'worker-timeout-capped in the full matrix. ANTs '
                            'SyNOnly is fast (~6 s at 128^3) -- not slow.'),
    ),
    complexity=(
        'STEADY ~ levels x iters x n_steps x N: each iteration warps both '
        'images to the midpoint (two scaling-and-squaring SVF integrations), '
        'computes the LNCC force, smooths it (fluid) + the velocity '
        '(diffusion) -- two Gaussians/iter -- then a midpoint compose+invert '
        'at the end. The heaviest recipe to COMPILE (two velocity fields), '
        'but ANTs SyNOnly (the gold standard) is FAST on CPU (~0.5/2.9/6.0 s '
        'at 48/96/128^3 measured), so the GPU win is NOT a given -- it must '
        'clear the ~4x cost bar to count (measured in ECONOMIC.md, not '
        'assumed). HBM ~ 2 velocity fields + scaling-squaring intermediates '
        '(heaviest after demons). The size tier varies the volume + carries '
        'anisotropic (1x1x3) points.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
