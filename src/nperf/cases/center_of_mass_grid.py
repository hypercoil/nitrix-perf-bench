# -*- coding: utf-8 -*-
"""Tier-2 geometry: ``nitrix.geometry.center_of_mass_grid`` vs scipy / cupy.

Centre of mass of a dense weight volume, treating each cell's index as its
coordinate -- nitrix (jax) vs **``scipy.ndimage.center_of_mass``** (the
canonical array / medical-imaging reference; verified equal to ~1e-14 in fp64)
+ a CuPy GPU ref reimplementing the same index-weighted reduction, scored
against an fp64 oracle. Pure reduction, so memory-bound and GPU-pure. Ratio vs
nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.geometry import center_of_mass_grid

from ._base import BuiltPoint, Case, to_cupy
from ._geometry import cupy_com_grid, np_com_grid, scipy_com_grid, volume_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    W = volume_input((d, d, d), param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(W))
    ref = np_com_grid(W.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(W)
        return (W,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda w: center_of_mass_grid(w)),
        'scipy.ndimage.center_of_mass': ('scipy', scipy_com_grid()),  # floor
        'cupy.center_of_mass_grid': ('cupy', cupy_com_grid()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube edge): cost ~ d³ (one pass over the volume).
_SIZES = [32, 64, 96]

CASE = Case(
    name='center_of_mass_grid',
    op_qualname='nitrix.geometry.center_of_mass_grid',
    output_independent=False,  # a centroid reduction over the whole volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 64, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
