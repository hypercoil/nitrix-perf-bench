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
tradeoff is a legitimate signal, not a row to drop.

**Scale-gaming defence (the size tier).**  A perf win at a small benched size
says nothing at brain scale when the *asymptotics* differ.  nitrix's separable
min-plus matmul does more FLOPs than scipy/cupy's Felzenszwalb-Huttenlocher EDT
(O(n^(d+1)) per axis vs O(n^d)) but in one **shallow** pass, where F-H is a
deeper data-dependent sequential scan; the working hypothesis is that GPU
wall-clock at small scale is bound by algorithmic *depth*, not FLOPs, so the
low-depth brute force wins there -- and loses once the FLOP + HBM cost
dominates at scale (nitrix materialises O(n^d) buffers, a 5-1000x HBM mult on
L4, vs F-H's in-place memory).  That is the crossover this case is built to
surface, not hide.  So the dev ``param_points`` stay small (the
drift/representative anchor) and the brain-scale sizes -- single MRI volumes to
256^3 / 512^2 *and a cohort batch sweep* -- live in ``large_param_points`` (run
by default; ``--skip-large`` drops them for fast dev cycles, stamping the run
non-authoritative).  ``tools/scaling_report.py`` reads the resulting curve to
surface the speed crossover, the HBM growth / projected OOM, and the stated
``complexity`` law.  The honest headline: nitrix EDT is a small-scale GPU win
(differentiability is a substrate bonus), not an at-cohort-scale one.  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import distance_transform

from ._base import ApproxBaseline, BuiltPoint, Case, to_cupy
from ._distance import (
    blob_mask,
    blob_stack,
    cupy_edt,
    cupy_edt_batched,
    scipy_edt,
    scipy_edt_batched,
)
from ._itk import sitk_edt


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    batched = bool(param.get('batch'))
    seed = param.get('seed', 0)

    if batched:
        # Batched (cohort) regime: EDT is all-axes-spatial, so a stack is
        # transformed per-image via vmap (the supported batch contract); the
        # references loop.  This is the axis where nitrix's per-volume HBM cost
        # compounds toward an OOM the single-volume sweep never reaches.
        mask = blob_stack(param['batch'], shape, seed)
        ref = scipy_edt_batched(mask)
        scipy_fn = scipy_edt_batched
        cupy_fn = cupy_edt_batched()
    else:
        mask = blob_mask(shape, seed)  # structured, real distances
        ref = scipy_edt(mask)  # exact EDT (fp64 oracle)
        scipy_fn = scipy_edt
        cupy_fn = cupy_edt()

    def nitrix_fn(m: Any) -> Any:
        # ``batched`` is a Python bool (static at trace time): vmap per-image
        # on a cohort stack, else the plain default call.
        if batched:
            return jax.vmap(distance_transform)(m)
        return distance_transform(m)

    jx = jax.block_until_ready(jnp.asarray(mask))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (jx,)

    baselines = {
        # default call (no metric) -- the euclidean engine users actually hit.
        'nitrix-jax': ('jax', nitrix_fn),
        'scipy.ndimage.distance_transform_edt': ('scipy', scipy_fn),
        'cupyx.scipy.ndimage.distance_transform_edt': (
            'cupy', cupy_fn),  # GPU on-target ref (exact EDT)
    }
    if not batched:
        # SimpleITK Danielsson -- declared-approximate (4SED, ~0.9 voxel of
        # exact); fidelity reported, not gated (see approximate_baselines).
        # Single-image only (no batched ITK path here).
        baselines['simpleitk.DanielssonDistanceMap'] = ('simpleitk', sitk_edt)
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# Dev tier (small, fast -- drift/dev anchor; representative = 64^3).
_SMALL = [[64, 64], [128, 128], [64, 64, 64]]
# Brain-scale size tier (large_param_points): single MRI-volume sizes where
# the O(n^(d+1))-per-axis matmul / HBM growth shows, plus a **cohort batch**
# sweep at fixed spatial size to isolate the per-subject HBM slope (-> OOM).
_LARGE = [[256, 256], [512, 512], [128, 128, 128], [256, 256, 256]]
_BATCHED = [(4, [128, 128, 128]), (8, [128, 128, 128]), (16, [128, 128, 128])]

CASE = Case(
    name='distance_transform',
    op_qualname='nitrix.morphology.distance_transform',
    output_independent=False,  # each output is a global min over the mask
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SMALL],
    large_param_points=tuple(
        [{'shape': s, 'seed': 0} for s in _LARGE]
        + [{'shape': s, 'batch': b, 'seed': 0} for b, s in _BATCHED]
    ),
    representative={'shape': [64, 64, 64], 'seed': 0},
    # The cost law (warranted, derived) + the working hypothesis for the
    # crossover.  FLOPs: per axis nitrix runs a (n^(d-1), n) x (n, n) min-plus
    # matmul = O(n^(d+1)); scipy/cupy use Felzenszwalb-Huttenlocher EDT at
    # O(n^d) -- nitrix does ~n x more work per axis.  *Depth*: nitrix's pass is
    # one **shallow** matmul; F-H is a deeper data-dependent sequential scan.
    # Hypothesis (unconfirmed): on the highly-parallel GPU, wall-clock at small
    # scale is bound by algorithmic *depth*, not FLOPs, so the low-depth brute
    # force wins there despite the extra work; at large / batched scale the
    # FLOP + HBM cost dominates and F-H wins (nitrix materialises O(n^d)
    # min-plus buffers, ~5-1000x the in-place F-H refs on the L4, so the cohort
    # batch OOMs first).  Differentiability is a *bonus* of the semiring
    # substrate, not the reason it was chosen.
    complexity=(
        'time nitrix O(n^(d+1))/axis (one shallow min-plus matmul) vs F-H '
        'O(n^d) (deeper sequential scan); HBM nitrix ~5-1000x the in-place '
        'F-H refs (L4). Hypothesis: GPU wall-clock depth-bound at small scale '
        '(low-depth brute force wins despite more FLOPs), flop/HBM-bound at '
        'large/batched scale (F-H wins, nitrix OOMs first). Differentiability '
        'is a bonus of the substrate, not the reason it was chosen'
    ),
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
