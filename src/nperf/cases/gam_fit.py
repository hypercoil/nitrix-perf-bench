# -*- coding: utf-8 -*-
"""Tier-2 HEADLINE: ``nitrix.stats.gam.gam_fit`` vs R mgcv::gam.

The mass-univariate penalised-spline GAM: per voxel fit ``y ~ s(x)`` (a
B-spline / P-spline smooth) with REML smoothing-parameter selection -- the
workhorse for voxelwise nonlinear effects (age curves, dose-response, ...).
**nitrix fits all ``V`` voxels in ONE batched penalised-IRLS call** (vmap over
the voxel batch behind one compile, each with its own selected lambda), while
the gold standard **R ``mgcv::gam``** (P-spline ``bs='ps'``, ``method='REML'``)
LOOPS one iterative fit per voxel -- the batched-vs-looped story (cf. glm_fit /
reml_fit), the gap growing with ``V``.

Fidelity: the **fitted smooth ``yhat``** is the convention-robust quantity (the
basis parametrisation differs between nitrix and mgcv, but the fitted curve
does not).  There is no exact cross-tool oracle (REML lambda selection differs
infinitesimally), so ``fp64_reference=None``; correctness is **agreement with
mgcv** -- VERIFIED the fitted smooths agree to ~2e-6 (both P-spline + REML),
validated out-of-band by correlation + recovery of the planted truth.  The
penalised-IRLS *core* is gated tightly against a numpy fixed-lambda
penalised-LS oracle in the tests.  R mgcv is CPU-only, looped, file-coupled ->
a ``slow_baseline`` with an ``R.iofloor`` (CSV + R startup, subtracted).
Keyed ``{V, N, n_basis}`` (scale axis = ``V``).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.basis import bspline_basis
from nitrix.stats.gam import gam_fit

from ._base import ApproxBaseline, BuiltPoint, Case, SlowBaseline
from ._gam import gam_data, r_mgcv_gam, r_mgcv_iofloor


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, big_n, k = int(param['V']), int(param['N']), int(param['n_basis'])
    x, Y = gam_data(v, big_n, param.get('seed', 0))
    sb = bspline_basis(jnp.asarray(x), k, center=True)
    # the fit design D = [intercept | centred basis]; yhat = D @ coef^T.
    big_d = jax.block_until_ready(jnp.concatenate(
        [jnp.ones((big_n, 1), sb.design.dtype), sb.design], axis=1))
    jY = jax.block_until_ready(jnp.asarray(Y))

    def _nitrix(y: Any) -> Any:  # batched GAM over V voxels -> fitted yhat
        return (big_d @ gam_fit(y, [sb]).coef.T).T

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y, x)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'R.mgcv': ('r', r_mgcv_gam(k)),          # gold standard (looped)
        'R.iofloor': ('r', r_mgcv_iofloor(k)),   # CSV + R startup floor
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None, ratio_reference='nitrix-jax',
        fidelity_note=(
            'no exact cross-tool oracle (REML lambda selection differs '
            'infinitesimally); the fitted smooth yhat agrees with mgcv to '
            '~2e-6 (both P-spline + REML), validated out-of-band. The '
            'penalised-IRLS core is gated vs a numpy fixed-lambda oracle in '
            'the tests.'),
    )


# (voxels V, observations N, basis dim n_basis): V is the scale axis (the
# batched-vs-looped speedup grows with V); N/n_basis fixed.
_DEV = [(256, 150, 15), (1024, 150, 15)]
# Brain-voxel scale: V to where the looped mgcv (one iterative gam per voxel,
# ~tens of ms each) is infeasible (dropped via --skip-slow); nitrix batches it.
_LARGE = [(16384, 150, 15), (65536, 150, 15)]


def _p(v: int, big_n: int, k: int) -> Dict[str, Any]:
    return {'V': v, 'N': big_n, 'n_basis': k, 'seed': 0}


CASE = Case(
    name='gam_fit',
    op_qualname='nitrix.stats.gam.gam_fit',
    tier='marquee',
    output_independent=True,  # each voxel is an independent GAM fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[_p(*p) for p in _DEV],
    representative=_p(256, 150, 15),
    large_param_points=tuple(_p(*p) for p in _LARGE),
    complexity=(
        'batched penalised-IRLS GAM with per-voxel REML smoothing selection '
        'over V voxels: O(V * n_outer * (N k^2)). nitrix fits all V in ONE '
        'call; R mgcv LOOPS one iterative gam per voxel, so the '
        'batched-vs-looped speedup GROWS with V (the headline, and why mgcv '
        'is a slow_baseline). HBM ~ V * (N + k^2). Scale axis = V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    slow_baselines=(
        SlowBaseline(
            'R.mgcv',
            'per-voxel R mgcv::gam fits looped in one Rscript (~tens of '
            'ms/voxel); minutes at V>=1024, infeasible at the brain large '
            'tier. CPU-only; skip in dev cycles, run in the full matrix.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'R.iofloor',
            'no-op: returns zeros, so its rel_to_tol is meaningless -- the '
            'row times the CSV round-trip + R/mgcv startup R.mgcv pays '
            '(economic subtracts the same-namespace floor).'),
    ),
)
