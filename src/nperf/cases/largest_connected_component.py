# -*- coding: utf-8 -*-
"""Morphology: ``nitrix.morphology.largest_connected_component``.

The boolean mask of the single largest connected foreground region.  Because
the output is a **boolean** mask (not labels) it is permutation-invariant, so
scipy ``label`` + argmax-size is a clean **co-oracle** (exact, verified 0
mismatch), and a cupyx ``label``-based reference is the on-target GPU bar.  The
input mask has one dominant component so the largest is unambiguous (a tie
would be a fidelity miss, recorded honestly).

Composes ``connected_components`` (iterative label propagation) + an
argmax-over-sizes.  Global op, GPU-pure.  Scale tier.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import largest_connected_component

from ._base import BuiltPoint, Case, to_cupy
from ._connectivity import blob_mask, cupyx_largest_cc, scipy_largest_cc


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    mask = blob_mask((d, d, d), param.get('seed', 0))
    mj = jax.block_until_ready(jnp.asarray(mask))
    ref = scipy_largest_cc(mask).astype('float64')  # fp64 co-oracle (boolean)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (mj,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: largest_connected_component(m)),
        'scipy.largest_cc': ('scipy', scipy_largest_cc),  # co-oracle / floor
        'cupy.largest_cc': ('cupy', cupyx_largest_cc()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='largest_connected_component',
    op_qualname='nitrix.morphology.largest_connected_component',
    output_independent=False,  # a global connectivity reduction
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'connected_components (pointer-jumping label propagation, O(log d) '
        'passes for diameter d) + a bincount/argmax over labels. Global, '
        'GPU-pure; steady ~ N log(d). MEASURED (L4, 48^3): like '
        'connected_components, nitrix lags cupyx label here (cupyx ~2x) -- a '
        'shared kernel/algorithm candidate. The size tier varies N.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
