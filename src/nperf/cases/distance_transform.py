# -*- coding: utf-8 -*-
"""Tier-2 morphology: ``nitrix.morphology.distance_transform`` (euclidean).

The **default, exact** Euclidean distance transform (B18 Win 1): nitrix
computes it as a separable per-axis tropical (min, +) matmul
``out[p] = min_q (g[q] + (q-p)^2)`` on the semiring kernel -- the metric
``scipy.ndimage.distance_transform_edt`` computes, matching it to fp32
round-off (~4e-6 abs on realistic distances).  This case measures the op the
way users call it (no ``metric`` kwarg -> the euclidean default) and gates it
**tight** (``atol=1e-4``): the historical ``atol=1.0`` was a crutch for the old
quasi-Euclidean chamfer default and would now hide an exact->approximate
regression (a chamfer fallback's ~0.4-voxel error fails this gate by ~4000x).

The chamfer engine (``metric='chebyshev'`` / ``'city_block'``) is a
**separate** case (``distance_transform_chamfer``) -- a different branch with a
different oracle.

Exact references: ``scipy.ndimage.distance_transform_edt`` (fp64 oracle + CPU
floor) and ``cupyx.scipy.ndimage.distance_transform_edt`` (on-target GPU ref).
SimpleITK's Danielsson is kept as a **declared-approximate** baseline: the
tight gate revealed it is the ~0.9-voxel-approximate 4SED algorithm, not an
exact EDT (the old ``atol=1.0`` + tiny random-mask distances had hidden that),
so its fidelity is *reported, not gated* -- a 4SED-vs-exact accuracy/speed
tradeoff is a legitimate signal, not a row to drop.  The size sweep runs to
256^3 / 512^2 so the O(n^2)-per-axis matmul crossover vs the O(n) separable
references is visible rather than hidden behind one small size.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import distance_transform

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._distance import blob_mask, cupy_edt, scipy_edt
from ._itk import sitk_edt


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    mask = blob_mask(shape, param.get('seed', 0))  # structured, real distances
    jx = jax.block_until_ready(jnp.asarray(mask))

    ref = scipy_edt(mask)  # exact EDT (fp64 oracle)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (jx,)

    baselines = {
        # default call (no metric) -- the euclidean engine users actually hit.
        'nitrix-jax': ('jax', lambda m: distance_transform(m)),
        'scipy.ndimage.distance_transform_edt': ('scipy', scipy_edt),
        # SimpleITK Danielsson -- declared-approximate (4SED, ~0.9 voxel of
        # exact); fidelity reported, not gated (see approximate_baselines).
        'simpleitk.DanielssonDistanceMap': ('simpleitk', sitk_edt),
        'cupyx.scipy.ndimage.distance_transform_edt': (
            'cupy', cupy_edt()),  # GPU on-target ref (exact EDT)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# 2-D and 3-D; sized up to 256^3 / 512^2 so the per-axis O(n^2) min-plus matmul
# crossover vs the O(n) separable EDT references is visible (B18 Win 1).
_SHAPES = [[256, 256], [512, 512], [64, 64, 64], [128, 128, 128],
           [256, 256, 256]]

CASE = Case(
    name='distance_transform',
    op_qualname='nitrix.morphology.distance_transform',
    output_independent=False,  # each output is a global min over the mask
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'seed': 0},
    build=_build,
    # Danielsson is the 4SED algorithm (~0.9-voxel max error vs exact EDT on
    # structured masks, measured on the L4); its fidelity is reported, not
    # gated -- a 4SED-vs-exact accuracy/speed tradeoff is the signal.
    approximate_baselines=(
        ApproxBaseline(
            'simpleitk.DanielssonDistanceMap',
            '4SED approximate EDT (~0.9 voxel max vs exact on blob masks, '
            'L4); reported not gated -- accuracy/speed tradeoff signal'),
    ),
    # tight: nitrix euclidean is now EXACT (matches scipy EDT to fp32
    # round-off, ~4e-6 abs on realistic distances); this gate fails an
    # exact->approximate regression (~0.4-voxel chamfer fallback) by ~4000x.
    rtol=1e-3,
    atol=1e-4,
)
