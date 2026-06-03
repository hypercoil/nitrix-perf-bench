# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.displacement_from_reference_grid``.

The centre-of-mass displacement from a fixed reference coordinate -- the
registration regulariser pattern ``center_of_mass_grid(W) - reference`` --
nitrix (jax) vs ``scipy.ndimage.center_of_mass`` minus the same reference (CPU
floor; the centroid is the canonical part, verified ~1e-14 in fp64) + a CuPy
GPU ref. The reference is the grid centre (a fixed parameter, baked outside the
timed region). Pure reduction, GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.geometry import displacement_from_reference_grid

from ._base import BuiltPoint, Case, to_cupy
from ._geometry import (
    cupy_displacement_grid,
    np_com_grid,
    scipy_displacement_grid,
    volume_input,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    W = volume_input((d, d, d), param.get('seed', 0))
    reference = ((np.array([d, d, d]) - 1) / 2.0).astype(np.float64)
    jx = jax.block_until_ready(jnp.asarray(W))
    jref = jax.block_until_ready(jnp.asarray(reference))
    ref = np_com_grid(W.astype('float64')) - reference  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W)
        return (W,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': (
            'jax', lambda w: displacement_from_reference_grid(w, jref)),
        'scipy.ndimage.center_of_mass': (
            'scipy', scipy_displacement_grid(reference)),  # CPU floor
        'cupy.displacement_from_reference_grid': (
            'cupy', cupy_displacement_grid(reference)),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube edge): cost ~ d³ (one pass over the volume).
_SIZES = [32, 64, 96]

CASE = Case(
    name='displacement_from_reference_grid',
    op_qualname='nitrix.geometry.displacement_from_reference_grid',
    output_independent=False,  # a centroid reduction over the whole volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 64, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
