# -*- coding: utf-8 -*-
"""Registration recipe (R1): ``nitrix.register.affine_register`` -- the driver.

The affine sibling of ``rigid_register`` (12-DOF in 3D via ``affine_exp`` ->
``matrix_exp``, coarse-to-fine GN/LM on SSD).  Same task-level framing (no
shared oracle -- ``fp64_reference`` is ``None``; recovery pinned in the tests).

**Post loop-roll (nitrix ``ddc2e10``): steady is the headline.**  Like rigid,
the optimiser loop is now a ``lax.scan``, so the cold compile is ~flat in
iterations and small (~4-11 s on the L4, was 24-211 s unrolled).  Two affine
specifics: (1) each LM iteration also evaluates ``matrix_exp`` (linear-block
exp) and assembles a P=12 normal system (vs rigid's P=6), so affine steady is
~the rigid cost at ~2x the per-iteration constant; (2) the ``L3x30`` *CPU*
compile that previously failed XLA outright (``INTERNAL: failed to materialize
symbols`` -- the unrolled graph was too large) **now compiles** (~6.6 s) -- the
loop-roll resolved it (nitrix recorded ``affine CPU-compile resolved``).

Two task-level domain references (CPU, not jit-compiled -> wall-clock with no
separate compile, read against nitrix's steady + one-time compile): ANTsPy
``registration(Affine)`` (fixed internal schedule) and **dipy** affine (12-DOF
mutual information, pyramid driven by this case's ``levels`` x ``iters`` -- see
``cases/_register.py``).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import (
    IndexSpace,
    RegistrationSpec,
    WorldSpace,
    affine_register,
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
        # REAL anatomy: the MNI152 T1 under a planted affine warp (same grid).
        # Affine recovers fine here (99^3 >= 32^3 -- the v3 small-grid
        # divergence is not in play; FR register-affine-small-grid-divergence).
        moving, fixed = real_warp_pair(int(param.get('resolution', 2)), seed)
        space = IndexSpace()
        ants_ref = ants_register('Affine')
        dipy_ref = dipy_register('affine', levels, iters)
    elif param.get('space') == 'world':
        # cross-grid: fixed/moving on different grids + anisotropic moving
        # spacing, recovered in physical space via WorldSpace; refs get the
        # matching spacing/affines (their native regime).
        shape = tuple(param['shape'])
        f_sp = tuple(param.get('fixed_spacing', (1.0, 1.0, 1.0)))
        m_sp = tuple(param.get('moving_spacing', (1.2, 1.0, 0.9)))
        moving, fixed, a_m, a_f = warp_pair_cross_grid(
            shape, tuple(param['moving_shape']),
            fixed_spacing=f_sp, moving_spacing=m_sp, seed=seed)
        space = WorldSpace(fixed_affine=jnp.asarray(a_f),
                           moving_affine=jnp.asarray(a_m))
        ants_ref = ants_register('Affine', spacing=(f_sp, m_sp))
        dipy_ref = dipy_register('affine', levels, iters, affines=(a_f, a_m))
    else:
        moving, fixed = warp_pair(tuple(param['shape']), seed)
        space = IndexSpace()  # the default (shared-grid) path, unchanged
        ants_ref = ants_register('Affine')
        dipy_ref = dipy_register('affine', levels, iters)
    mj = jax.block_until_ready(jnp.asarray(moving))
    fj = jax.block_until_ready(jnp.asarray(fixed))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (mj, fj)
        return (moving, fixed)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda mv, fx: affine_register(mv, fx, spec=spec,
                                           space=space).params),
        'ants.registration': ('ants', ants_ref),   # ANTs Affine (physical sp.)
        'dipy.registration': ('dipy', dipy_ref),    # dipy affine MI
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level recipe: no shared oracle (nitrix GN/LM, '
                       'ANTs, dipy converge to different transforms). Reads '
                       'steady_time (the headline post loop-roll) + the '
                       'one-time compile; recovery pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


_CONFIGS = [(1, 10), (2, 20), (3, 30)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): fix a mid config, vary the volume. affine carries
# the P=12 normal system + matrix_exp/iter but the same ~image-sized HBM as
# rigid (the assembled J is P-thin), so it too keeps headroom to 192^3.
_LARGE = [[96, 96, 96], [128, 128, 128], [160, 160, 160], [192, 192, 192]]
# Cross-grid points (cross-resolution/cross-modal): fixed/moving on different
# grids + anisotropic moving spacing, recovered via WorldSpace (alongside the
# shared-grid points; the refs' native physical-space regime).
_LARGE_WORLD = [
    {'shape': [96, 96, 96], 'moving_shape': [80, 96, 112],
     'levels': 2, 'iters': 20, 'seed': 0, 'space': 'world',
     'fixed_spacing': [1, 1, 1], 'moving_spacing': [1.2, 1.0, 0.9]},
    {'shape': [128, 128, 128], 'moving_shape': [112, 128, 144],
     'levels': 2, 'iters': 20, 'seed': 0, 'space': 'world',
     'fixed_spacing': [1, 1, 1], 'moving_spacing': [1.2, 1.0, 0.9]},
]
# Real-anatomy point: MNI152 T1 (~99^3 @2mm) under a planted affine warp.
_LARGE_REAL = [{'data': 'mni152', 'resolution': 2, 'levels': 2, 'iters': 20,
                'seed': 0}]

CASE = Case(
    name='affine_register',
    op_qualname='nitrix.register.affine_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 10, 'seed': 0},
    large_param_points=tuple(
        [{'shape': s, 'levels': 2, 'iters': 20, 'seed': 0} for s in _LARGE]
        + _LARGE_WORLD + _LARGE_REAL),
    # dipy MI is slow at scale (128^3 ~25 s on CPU); skippable for dev cycles.
    slow_baselines=(SlowBaseline(
        'dipy.registration',
        reason='dipy MI ~25 s at 128^3 on CPU (CPU-only cython); '
               'worker-timeout-capped in the full matrix.'),),
    complexity=(
        'post loop-roll (lax.scan): COMPILE ~flat in iterations, ~4-11 s (was '
        '24-211 s unrolled; the L3x30 CPU compile that failed XLA now '
        'compiles). STEADY ~ iterations x P x N with P=12 (assemble J^TJ + a '
        'matrix_exp of the linear block + a P x P solve) -- ~2x rigid '
        'per-iter. GPU steady is overhead-bound below ~48^3 then '
        'compute-bound; the GPU/CPU speedup climbs to a brain-scale plateau '
        '~35x. HBM like rigid (J is P-thin); cold peak_hbm is '
        'autotune-contaminated -- no OOM projection (see '
        'reports/REGISTRATION_SCALING.md). Bias: fixed (levels=2, iters=20); '
        'real pipelines raise levels with resolution.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
