# -*- coding: utf-8 -*-
"""Registration recipe (R1): ``nitrix.register.affine_register`` -- the driver.

The affine sibling of ``rigid_register`` (12-DOF in 3D via ``affine_exp`` ->
``matrix_exp``, coarse-to-fine GN/LM on SSD).  Same task-level framing (no
shared oracle -- ``fp64_reference`` is ``None``; recovery pinned in the tests)
and the **same compile-vs-steady headline**: the optimizer loop is
Python-unrolled, so the XLA cold compile scales with the total unrolled
iteration count and dominates first-call latency.  Affine also evaluates
``matrix_exp`` (the linear-block exp) inside each iteration's linearised
Jacobian, so its per-iteration graph is a little heavier than rigid's.

ANTsPy ``registration(type_of_transform='Affine')`` is the task-level domain
reference (CPU; not jit-compiled -> wall-clock with no separate compile, read
against nitrix's steady + one-time compile).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import RegistrationSpec, affine_register

from ._base import BuiltPoint, Case
from ._register import ants_register, warp_pair


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
        'ants.registration': (  # task-level domain ref
            'ants', ants_register('Affine')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level recipe: no shared oracle (nitrix GN/LM and '
                       'ANTs converge to different transforms). Headline is '
                       'compile_time vs steady_time (the cold-compile '
                       'diagnosis); recovery accuracy pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


_CONFIGS = [(1, 10), (2, 20), (3, 30)]
_SHAPE = [48, 48, 48]

CASE = Case(
    name='affine_register',
    op_qualname='nitrix.register.affine_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 10, 'seed': 0},
    complexity=(
        'same cold-compile story as rigid_register: compile scales ~linearly '
        'with the total unrolled iters (levels x iterations) because the '
        'optimizer loop is Python-unrolled -- first-call latency, not steady '
        'state. Like rigid, each outer iteration runs an inner CG solve whose '
        'matvecs are a matrix-free autodiff Jacobian (the dominant '
        'per-iteration cost; demons, with no inner solve, is ~5x cheaper per '
        'iter) -- and affine also runs matrix_exp (the linear-block exp) '
        'inside each linearised iteration, so its per-iter graph is a bit '
        'heavier than rigid. Fix: roll the loop (lax.scan) + closed-form '
        'affine Jacobian.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
