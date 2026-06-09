# -*- coding: utf-8 -*-
"""Registration recipe (R2): ``nitrix.register.diffeomorphic_demons_register``.

Diffeomorphic log-Demons (stationary velocity field; ESM force, Gaussian
fluid+diffusion regularisation, scaling-and-squaring exponential).  The
non-rigid sibling of the affine recipes, and the same task-level framing (no
shared oracle -- ``fp64_reference`` is ``None``; recovery pinned in the tests).

**Post loop-roll (nitrix ``ddc2e10``): steady is the headline, and HBM is the
binding constraint at scale.**  The Demons iterations + SVF integration are now
a ``lax.scan``, so the cold compile is **flat in iterations** -- ``L2x20`` and
``L2x40`` compile *identically* (~6.8 s on the L4); even the default ``L3x80``
(240 iters), once minutes to compile when unrolled, is now ~7 s.  The
per-iteration work (no inner solve): an ESM force, two ``spatial_gradient`` s,
``n_steps`` scaling-squaring warps, two Gaussian smooths.  Per iteration demons
stays cheaper than rigid/affine GN/LM, but the gap narrowed (those now
assemble a small-P normal system rather than a matrix-free autodiff-Jacobian
CG).  What demons pays instead is **memory**: the d-component velocity field +
scaling-squaring intermediates make its per-voxel HBM ~1.7x rigid/affine, so it
hits the device ceiling first (see ``complexity`` / the size tier).

Two diffeomorphic task-level domain references (CPU, not jit-compiled): ANTsPy
``registration(SyNOnly)`` -- pure SyN deformable, no rigid/affine pre-step, the
counterpart of nitrix's pure log-Demons (the default ``'SyN'`` preset, which
prepends rigid+affine, fails on this pair) -- and **dipy**
``SymmetricDiffeomorphicRegistration`` on SSD (the counterpart of nitrix's
SSD-driven log-Demons; pyramid driven by this case's ``levels`` x ``iters`` --
see ``cases/_register.py``).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import DemonsSpec, diffeomorphic_demons_register

from ._base import BuiltPoint, Case
from ._register import ants_register, dipy_register, warp_pair


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
        # full scan to run so compile_time is the real cold compile).
        'nitrix-jax': (
            'jax',
            lambda mv, fx: diffeomorphic_demons_register(
                mv, fx, spec=spec).velocity),
        'ants.registration': (  # diffeomorphic task-level domain ref (SyNOnly)
            'ants', ants_register('SyNOnly')),
        'dipy.registration': (  # diffeomorphic ref (dipy SyN on SSD)
            'dipy', dipy_register('syn', int(param['levels']),
                                  int(param['iters']))),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level: no shared oracle (nitrix log-Demons, ANTs '
                       'SyNOnly, dipy SyN converge to different warps). '
                       'Reads steady_time (the headline post loop-roll) + the '
                       'one-time compile; recovery pinned in the tests.'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: total iters 20/40/80; with the loop rolled these now compile in
# ~seconds (and L2x20 == L2x40 in compile -- flat in iters).
_CONFIGS = [(1, 20), (2, 20), (2, 40)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): fix a mid config, vary the volume. Capped at 160^3:
# demons' SVF field + scaling-squaring intermediates cost ~3 KB/voxel (~1.7x
# rigid/affine), so it is the HBM-bound recipe -- scaling_report projects OOM
# ~187^3 on the L4 from this curve (the binding brain-scale constraint).
_LARGE = [[96, 96, 96], [128, 128, 128], [160, 160, 160]]

CASE = Case(
    name='diffeomorphic_demons',
    op_qualname='nitrix.register.diffeomorphic_demons_register',
    output_independent=False,  # a global iterative fit over the whole image
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 20, 'seed': 0},
    large_param_points=tuple(
        {'shape': s, 'levels': 2, 'iters': 20, 'seed': 0} for s in _LARGE),
    complexity=(
        'post loop-roll (lax.scan): COMPILE flat in iterations -- L2x20 == '
        'L2x40 (~6.8 s on the L4); even the default L3x80 (240 iters), once '
        'minutes unrolled, is ~7 s. STEADY ~ iterations x n_steps x N (ESM '
        'force + 2 spatial_gradients + n_steps scaling-squaring warps + 2 '
        'Gaussians; no inner solve), but SUPER-linear at large N (bandwidth-'
        'bound on the SVF field): the GPU/CPU speedup peaks ~43x (48-96^3) '
        'then erodes to ~28x (160^3) -- the most bandwidth-bound recipe at '
        'scale. HBM: the heaviest recipe (~3 vs rigid/affine ~1.8 KB/voxel at '
        'clean small sizes), but cold peak_hbm is contaminated by XLA '
        'autotune scratch (a shared ~8.7 GB 128^3 spike, non-monotonic) so NO '
        'OOM projection is trustworthy; none hit OOM to 160^3 on the 23 GB '
        'L4. See reports/REGISTRATION_SCALING.md.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
