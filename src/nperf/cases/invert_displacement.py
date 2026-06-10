# -*- coding: utf-8 -*-
"""Geometry: ``nitrix.geometry.invert_displacement`` (iterative + IFT-diff).

The inverse displacement field of ``φ = id + s`` -- the fixed point
``s_inv = -s ∘ (id + s_inv)`` (``numerics.fixed_point_solve``).  Two things
make this case special:

- **The unique win is differentiability.** nitrix's fixed-point solver is a
  ``jax.custom_vjp``: it differentiates through the *solution* by the
  implicit-function theorem (the adjoint is itself a fixed-point solve), so
  ``invert_displacement`` is differentiable w.r.t. ``s``.  numpy / scipy / cupy
  have **no gradient** through the iteration at all -- the eigensolver story
  ([[the lobpcg -vjp]]): the adjoint, not the algorithm, is the capability.
- **Early-exit done right (the iterative-op caveat, the *good* case).** The
  forward is a ``lax.while_loop`` with a relative-``tol`` convergence test (not
  a fixed ``lax.scan``), so it is *difficulty-adaptive* -- the exact lever the
  registration-recipe optimisers lack and the filed
  ``registration-early-stopping-while-loop`` FR proposes for them. Here it
  already coexists with the implicit backward.

Warranted comparison: the numpy reimplementation runs the *same* Picard fixed
point to the *same* relative ``tol`` (iso-tolerance -- a fair iterative
comparison; verified ~7e-7 + a clean ``s_inv + s∘(id+s_inv) ≈ 0`` residual),
the fp64 oracle + CPU floor; cupy is the GPU bar (its per-iteration host
convergence check **syncs** -- the naive iterative reimpl nitrix's jitted,
fully-on-device ``while_loop`` improves on).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import invert_displacement

from ._base import BuiltPoint, Case, to_cupy
from ._registration import (
    cupy_invert_displacement,
    displacement_input,
    jacobian_sizes,
    np_invert_displacement,
)

_NDIM = 3


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    # small displacement (‖∇s‖ < 1) -- the diffeomorphic regime where the
    # fixed point converges (a foldless, invertible warp).
    s = displacement_input(jacobian_sizes(d), _NDIM, param.get('seed', 0),
                           scale=0.1)
    jx = jax.block_until_ready(jnp.asarray(s))
    ref = np_invert_displacement(s.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(s)
        return (s,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: invert_displacement(x)),
        'numpy.invert_displacement': ('numpy', np_invert_displacement),
        'cupy.invert_displacement': ('cupy', cupy_invert_displacement()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128]  # iterative + a 3-channel warp/iter -> capped vs the others

CASE = Case(
    name='invert_displacement',
    op_qualname='nitrix.geometry.invert_displacement',
    output_independent=False,  # each Picard step warps (couples) the field
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'O(K x N): K Picard iterations (data-adaptive -- a lax.while_loop to '
        'relative tol, NOT a fixed scan), each a 3-channel linear-interp warp '
        '(gather) over N voxels. Compile flat (one while_loop body); steady ~ '
        'K x N, K set by convergence (‖∇s‖ controls K). The headline is '
        'capability: IFT-differentiable (numpy/cupy are not) + early-exit '
        'coexisting with the implicit backward. The brain-scale tier varies N '
        '(capped at 128^3 -- iterative + the cupy host-sync ref).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
