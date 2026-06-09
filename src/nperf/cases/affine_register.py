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
from nitrix.register import RegistrationSpec, affine_register

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
        return (moving, fixed)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda mv, fx: affine_register(mv, fx, spec=spec).params),
        'ants.registration': (  # task-level domain ref (ANTs Affine)
            'ants', ants_register('Affine')),
        'dipy.registration': (  # task-level domain ref (dipy affine MI)
            'dipy', dipy_register('affine', int(param['levels']),
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


_CONFIGS = [(1, 10), (2, 20), (3, 30)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): fix a mid config, vary the volume. affine carries
# the P=12 normal system + matrix_exp/iter but the same ~image-sized HBM as
# rigid (the assembled J is P-thin), so it too keeps headroom to 192^3.
_LARGE = [[96, 96, 96], [128, 128, 128], [160, 160, 160], [192, 192, 192]]

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
        {'shape': s, 'levels': 2, 'iters': 20, 'seed': 0} for s in _LARGE),
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
