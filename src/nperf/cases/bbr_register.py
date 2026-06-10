# -*- coding: utf-8 -*-
"""Registration recipe (R4 v2): ``nitrix.register.bbr_register`` -- BBR.

**Boundary-based registration** (Greve-Fischl): a rigid transform refined by
maximising intensity contrast across a tissue boundary (the canonical EPI ->
structural alignment).  Driven by a surface (``points`` + outward ``normals``),
not an image pair; BFGS over the rigid parameters, each cost evaluation
sampling ``2N`` points along the normals.

**No ITK / ANTs counterpart.**  Boundary-based registration is **not** in ITK
or ANTs -- it is FSL ``flirt -bbr`` / FreeSurfer ``bbregister`` territory,
neither
installed (a planned ``/scratch`` install, revisit with the volreg community
tools).  So this case is **nitrix-only**: it reads nitrix **GPU vs CPU** + the
one-time compile (the economic verdict falls back to GPU-vs-own-CPU -- "is the
GPU worth the ~4x hardware premium for *this* op", with no domain tool yet).
``fp64_reference`` is ``None``; the recovery (the planted boundary offset is
seated back -- ``cost_history[-1] < cost_history[0]``) is pinned in the tests.

The cost is driven by ``N`` boundary points x BFGS ``iterations`` and is
**volume-independent** (only ``2N`` samples touch the grid), so the size tier
sweeps ``N`` to cortical-mesh scale (a hemisphere surface is ~150k vertices).
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import BBRSpec, bbr_register

from ._base import BuiltPoint, Case
from ._register import bbr_boundary


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    moving, points, normals = bbr_boundary(shape, int(param['N']),
                                           param.get('seed', 0))
    spec = BBRSpec(iterations=int(param['iters']))
    mj = jax.block_until_ready(jnp.asarray(moving))
    pj = jax.block_until_ready(jnp.asarray(points))
    nj = jax.block_until_ready(jnp.asarray(normals))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (mj, pj, nj)  # nitrix-only: every baseline is jax

    baselines = {
        # return the optimised rigid params (the deliverable): forces the full
        # BFGS over the boundary cost to run.
        'nitrix-jax': (
            'jax', lambda mv, pts, nrm: bbr_register(mv, pts, nrm,
                                                     spec=spec).params),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('nitrix-only: boundary-based registration has NO '
                       'ITK/ANTs counterpart (FSL/FreeSurfer territory, not '
                       'installed). Reads GPU vs CPU + the one-time compile; '
                       'recovery (cost_history[-1] < [0]) pinned in the '
                       'tests. Economic verdict is GPU-vs-own-CPU pending a '
                       'community BBR tool.'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: vary the BFGS iterations at a small surface / volume.
# Representative = the cheap (iters=50, N=2000) for drift / --quick.
_SHAPE = [48, 48, 48]
_DEV = [(50, 2000), (100, 2000)]
# Size tier: the cost is N x iters and VOLUME-INDEPENDENT, so sweep N to
# cortical-mesh scale (a hemisphere is ~150k vertices) at a fixed 64^3 grid.
_LARGE = [{'shape': [64, 64, 64], 'N': n, 'iters': 100, 'seed': 0}
          for n in (5_000, 20_000, 80_000)]

CASE = Case(
    name='bbr_register',
    op_qualname='nitrix.register.bbr_register',
    output_independent=False,  # a global BFGS fit over the rigid params
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'N': n, 'iters': it, 'seed': 0}
                  for (it, n) in _DEV],
    representative={'shape': _SHAPE, 'N': 2000, 'iters': 50, 'seed': 0},
    large_param_points=tuple(_LARGE),
    complexity=(
        'STEADY ~ iters x N: BFGS over the (rigid) parameters, each cost eval '
        'samples 2N points along the boundary normals + a tanh contrast -- '
        'VOLUME-INDEPENDENT (only 2N samples touch the grid). NO ITK/ANTs '
        'equivalent (a nitrix-only capability; the comparison is GPU vs CPU + '
        'the one-time compile, no domain tool). HBM ~ N (the point arrays), '
        'tiny. The size tier varies N to cortical-mesh scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
