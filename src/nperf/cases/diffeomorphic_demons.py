# -*- coding: utf-8 -*-
"""Registration recipe (R2): ``nitrix.register.diffeomorphic_demons_register``.

Diffeomorphic log-Demons (stationary velocity field; ESM force, Gaussian
fluid+diffusion regularisation, scaling-and-squaring exponential).  The
non-rigid sibling of the affine recipes, and the same task-level framing (no
shared oracle -- ``fp64_reference`` is ``None``; recovery pinned in the tests).

**Same compile-vs-steady headline.**  The Demons iterations + pyramid levels
are Python-**unrolled**, so the XLA cold compile scales with the total unrolled
iteration count -- and the default ``levels=3, iterations=80`` is **240
unrolled iterations**, the heaviest compile of the recipe family (minutes).
The per-iteration work differs from GN/LM (no inner solve): an ESM force, two
``spatial_gradient`` s, ``n_steps`` scaling-squaring warps, and two Gaussian
smooths -- still all unrolled.  Param points use modest configs (total iters
20/40/80) so the compile is benchable; the ``L3×80`` default is documented (it
would compile for minutes -- the same roll-the-loop fix applies).

ANTsPy ``registration(type_of_transform='SyNOnly')`` is the diffeomorphic
task-level domain reference -- pure SyN deformable (no rigid/affine pre-step),
the counterpart of nitrix's pure log-Demons (the default ``'SyN'`` preset,
which prepends rigid+affine, fails on this pair).  CPU; not jit-compiled.
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import DemonsSpec, diffeomorphic_demons_register

from ._base import BuiltPoint, Case
from ._register import ants_register, warp_pair


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    spec = DemonsSpec(levels=int(param['levels']),
                      iterations=int(param['iters']))
    moving, fixed = warp_pair(shape, param.get('seed', 0))
    mj = jax.block_until_ready(jnp.asarray(moving))
    fj = jax.block_until_ready(jnp.asarray(fixed))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (mj, fj)
        return (moving, fixed)

    baselines = {
        # the recipe; return velocity (the SVF parametrisation -- forces the
        # full unrolled loop so compile_time is the real cold compile).
        'nitrix-jax': (
            'jax',
            lambda mv, fx: diffeomorphic_demons_register(
                mv, fx, spec=spec).velocity),
        'ants.registration': (  # diffeomorphic task-level domain ref
            'ants', ants_register('SyNOnly')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level: no shared oracle (nitrix log-Demons '
                       'and ANTs SyNOnly converge to different deformations). '
                       'Headline is compile_time vs steady_time (cold '
                       'compile); recovery accuracy pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


# Modest configs (total iters 20/40/80) so the unrolled compile is benchable;
# the default L3x80 (240 iters) would compile for minutes -- documented.
_CONFIGS = [(1, 20), (2, 20), (2, 40)]
_SHAPE = [48, 48, 48]

CASE = Case(
    name='diffeomorphic_demons',
    op_qualname='nitrix.register.diffeomorphic_demons_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 20, 'seed': 0},
    complexity=(
        'same cold-compile story: the Demons iterations + pyramid levels are '
        'Python-unrolled, so compile scales ~linearly with total unrolled '
        'iters (levels x iterations) -- the default L3x80 is 240 iters, '
        'minutes to compile. Per-iteration: ESM force + 2 spatial_gradients + '
        'scaling-squaring warps + 2 Gaussians (NO inner solve) -- so steady '
        'is fast and ~5x cheaper per iteration than rigid/affine GN/LM at the '
        'same levels x iters (L2x20: 9 vs 45 ms), since those pay an inner CG '
        'solve of the autodiff Jacobian. First-call latency is the cost. '
        'Fix: roll the loop (lax.scan).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
