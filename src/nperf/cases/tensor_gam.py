# -*- coding: utf-8 -*-
"""Tier-2 GAM breadth: ``nitrix.stats.basis.tensor_product_basis`` (via
gam_fit) vs R mgcv ``te(x1, x2)``.

A penalised-spline GAM with a **tensor-product interaction** smooth over two
covariates (the 2-D surface ``f(x1, x2)``) -- the distinctive multi-margin
basis. Batched nitrix gam_fit over V voxels vs the gold standard R mgcv
(``te(x1, x2)``, REML, looped). ``n_basis`` is the PER-MARGIN basis dimension
(total ~ ``n_basis^2`` tensor columns). Fidelity = mgcv agreement on the fitted
surface (VERIFIED corr ~0.99 -- the te() reparametrisation differs more between
tools than the 1-D smooths, but the fitted surface agrees; no exact cross-tool
oracle -> ``fp64_reference=None``). Keyed ``{V, N, n_basis}`` (scale axis = V).
Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict

from ._base import ApproxBaseline, Case, SlowBaseline
from ._gam import build_smooth_case


def _p(v: int, big_n: int, k: int) -> Dict[str, Any]:
    return {'smooth': 'tensor', 'V': v, 'N': big_n, 'n_basis': k, 'seed': 0}


CASE = Case(
    name='tensor_gam',
    op_qualname='nitrix.stats.basis.tensor_product_basis',
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[_p(256, 200, 6), _p(1024, 200, 6)],
    representative=_p(256, 200, 6),
    large_param_points=(_p(16384, 200, 6),),
    complexity=(
        'batched penalised-spline GAM with a TENSOR-PRODUCT interaction '
        'smooth te(x1,x2) over V voxels (n_basis per margin -> ~n_basis^2 '
        'columns). nitrix batches all V; R mgcv te(x1,x2) LOOPS one fit per '
        'voxel -> the speedup grows with V. Scale axis = V.'),
    build=build_smooth_case,
    rtol=1e-3,
    atol=1e-4,
    slow_baselines=(
        SlowBaseline('R.mgcv',
                     'per-voxel mgcv tensor-product fits looped in one '
                     'Rscript; minutes at V>=1024, infeasible at the large '
                     'tier. CPU-only; skip in dev cycles.'),),
    approximate_baselines=(
        ApproxBaseline('R.iofloor',
                       'no-op: times the CSV + R/mgcv startup R.mgcv pays '
                       '(economic subtracts the same-namespace floor).'),),
)
