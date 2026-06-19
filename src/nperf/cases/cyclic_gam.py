# -*- coding: utf-8 -*-
"""Tier-2 GAM breadth: ``nitrix.stats.basis.cyclic_cubic_basis`` (via gam_fit)
vs R mgcv ``s(x, bs='cc')``.

A penalised-spline GAM with a **cyclic** smooth (periodic covariate -- phase,
time-of-day, angle), so the fit wraps at the boundary.  Batched nitrix gam_fit
over V voxels vs the gold standard R mgcv (cyclic cubic ``bs='cc'``, REML,
looped). Fidelity = mgcv agreement on the fitted smooth (VERIFIED corr
~0.9999; no exact cross-tool oracle -> ``fp64_reference=None``). Keyed
``{V, N, n_basis}`` (scale axis = V). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict

from ._base import ApproxBaseline, Case, SlowBaseline
from ._gam import build_smooth_case


def _p(v: int, big_n: int, k: int) -> Dict[str, Any]:
    return {'smooth': 'cyclic', 'V': v, 'N': big_n, 'n_basis': k, 'seed': 0}


CASE = Case(
    name='cyclic_gam',
    op_qualname='nitrix.stats.basis.cyclic_cubic_basis',
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[_p(256, 150, 12), _p(1024, 150, 12)],
    representative=_p(256, 150, 12),
    large_param_points=(_p(16384, 150, 12),),
    complexity=(
        'batched penalised-spline GAM with a CYCLIC smooth over V voxels '
        '(periodic covariate, wraps at the boundary). nitrix batches all V; '
        'R mgcv s(x,bs="cc") LOOPS one fit per voxel -> the speedup grows '
        'with V. Scale axis = V.'),
    build=build_smooth_case,
    rtol=1e-3,
    atol=1e-4,
    slow_baselines=(
        SlowBaseline('R.mgcv',
                     'per-voxel mgcv cyclic-spline fits looped in one '
                     'Rscript; minutes at V>=1024, infeasible at large tier. '
                     'CPU-only; skip in dev cycles.'),),
    approximate_baselines=(
        ApproxBaseline('R.iofloor',
                       'no-op: times the CSV + R/mgcv startup R.mgcv pays '
                       '(economic subtracts the same-namespace floor).'),),
)
