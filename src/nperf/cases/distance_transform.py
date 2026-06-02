# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.morphology.distance_transform`` vs scipy.

nitrix computes a quasi-Euclidean / iterative-tropical distance transform;
``scipy.ndimage.distance_transform_edt`` is the *exact* EDT.  The two are not
bit-identical, but nitrix's approximation error vs exact is **bounded and
size-independent** (~0.41 voxel in 2-D, ~0.73 in 3-D -- the diagonal-step
chamfer residual), so the exact EDT is a legitimate oracle with an
**op-appropriate tolerance**: ``atol = 1`` voxel verifies "within a voxel of
exact EDT" (its documented contract) while still catching real breakage
(> 1 voxel).  Ratio via ``--reference scipy.ndimage.distance_transform_edt``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.morphology import distance_transform

from ._base import BuiltPoint, Case, to_cupy
from ._itk import sitk_edt


def _cupy_distance_transform(m: Any) -> Any:
    '''GPU exact EDT (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).  Like
    scipy this is the *exact* EDT (not nitrix's quasi-Euclidean iterative DT),
    so it sits within the case's 1-voxel tolerance of the oracle.'''
    from cupyx.scipy import ndimage as cnd

    return cnd.distance_transform_edt(m > 0.5)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    rng = np.random.default_rng(param.get('seed', 0))
    mask = (rng.random(shape) > 0.5).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(mask))

    ref = spnd.distance_transform_edt(mask > 0.5)  # exact EDT (fp64 oracle)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: distance_transform(m)),
        'scipy.ndimage.distance_transform_edt': (
            'scipy', lambda m: spnd.distance_transform_edt(m > 0.5)),
        'simpleitk.DanielssonDistanceMap': (  # ITK floor (exact EDT, verified)
            'simpleitk', sitk_edt),
        'cupyx.scipy.ndimage.distance_transform_edt': (
            'cupy', _cupy_distance_transform),  # GPU on-target ref (exact EDT)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# 2-D and 3-D binary masks (the PERF_AUDIT ladder).
_SHAPES = [[32, 32], [128, 128], [32, 32, 32], [64, 64, 64]]

CASE = Case(
    name='distance_transform',
    op_qualname='nitrix.morphology.distance_transform',
    output_independent=False,  # each output is a global min over the mask
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    # 1-voxel tolerance: nitrix is an approximate (quasi-Euclidean) DT, exact
    # to ~1 voxel by construction; the gate still fails >1-voxel regressions.
    rtol=1e-2,
    atol=1.0,
)
