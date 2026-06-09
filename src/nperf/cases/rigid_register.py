# -*- coding: utf-8 -*-
"""Registration recipe (R1): ``nitrix.register.rigid_register`` -- the driver.

An **end-to-end** rigid registration (coarse-to-fine Gauss-Newton/LM on SSD).
Unlike the op-vs-oracle cases this is **task-level**: nitrix and ANTs converge
to different transforms, so there is **no shared oracle** (``fp64_reference``
is ``None``); recovery accuracy is pinned in ``tests/test_register_cases.py``,
and the bench reads the **time split**.

**The headline is ``compile_time`` vs ``steady_time``** -- the diagnosis of the
"registration slow on GPU" report.  nitrix's recipe is a **Python-unrolled**
fixed-iteration loop, so the XLA *cold compile* scales with the total unrolled
iteration count and dominates first-call latency, while the steady state is
fast.  Measured (L4, 48³): the default ``levels=3, iterations=30`` compiles
~**145 s** then runs ~**38 ms** steady; smaller configs compile faster
(``L1×10`` ~13 s, ``L2×20`` ~40 s) -- the unrolled-loop signature.  The suite
splits compile from steady, so the cold compile is a first-class number here
(filed with nitrix: ``registration-recipe-cold-compile`` -- roll the loop +
closed-form affine Jacobian).  Param points span the spec configs so the
compile scaling is visible; the representative is the fast ``L1×10`` so
``--quick`` / drift stay cheap.

Two task-level domain references (both own refs env, CPU, not jit-compiled, so
wall-clock is the full registration with no separate compile) -- read against
nitrix's steady + one-time compile: ANTsPy ``registration(Rigid)`` (fixed
internal schedule) and **dipy** rigid (mutual information, pyramid driven by
this case's ``levels`` x ``iters`` -- see ``cases/_register.py``).  Ratio vs
``nitrix-jax``.
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
        return (moving, fixed)  # numpy arrays for ANTs

    baselines = {
        # the recipe; return params (the deliverable -- forces the full
        # unrolled loop to run, so compile_time is the real cold compile).
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
        fidelity_note=('task-level recipe: no shared oracle (nitrix GN/LM and '
                       'ANTs converge to different transforms). Headline is '
                       'compile_time vs steady_time (the cold-compile '
                       'diagnosis); recovery accuracy pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


# Spec configs at one volume size, spanning the unrolled iteration count so the
# cold-compile scaling is visible. Representative = the fast L1x10 (cheap
# compile, for --quick / drift); the default L3x30 is the ~145 s-compile point.
_CONFIGS = [(1, 10), (2, 20), (3, 30)]
_SHAPE = [48, 48, 48]

CASE = Case(
    name='rigid_register',
    op_qualname='nitrix.register.rigid_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 10, 'seed': 0},
    complexity=(
        'cold compile scales ~linearly with the total unrolled iters '
        '(levels x iterations): the optimizer loop is Python-unrolled, so XLA '
        'compiles the whole graph -- ~145 s at the default L3x30 (vs ~38 ms '
        'steady) on the L4, the "slow on GPU". Each GN/LM outer iter runs '
        'an inner CG solve whose matvecs ARE a matrix-free autodiff Jacobian '
        '(jax.linearize through the warp, ~2 warp-passes each, no closed '
        'form) -- so an iteration ~= 10-20 warp-passes, which is why demons '
        '(no inner solve) is ~5x cheaper per iteration at the same budget, '
        'and inflates both steady cost and graph size. Steady state is fast; '
        'first-call latency is the cost. Fix: roll the loop (lax.scan) + a '
        'closed-form affine Jacobian.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
