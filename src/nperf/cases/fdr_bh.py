# -*- coding: utf-8 -*-
"""Tier-2 stats inference: ``nitrix.stats.inference.fdr_bh`` vs statsmodels.

Benjamini-Hochberg FDR correction of a vector of ``n`` p-values -> BH-adjusted
q-values (monotone non-decreasing in rank). The canonical neuro/stats community
tool is **statsmodels** ``multipletests(method='fdr_bh')`` (an exact CPU impl);
the fp64 oracle is an independent **numpy** reimpl (sort + rank-scale +
reverse-cummin; matches statsmodels exactly). nitrix runs it on-device behind
one compile (the whole-brain voxel vector); statsmodels sorts on the host.
``fdr_bh`` returns ``(rejected, p_adjusted)`` -- the case scores the continuous
``p_adjusted`` (the rejection mask is its thresholding at alpha). Keyed ``n`` =
the number of p-values (the scale axis = whole-brain voxel count). Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.inference import fdr_bh

from ._base import BuiltPoint, Case
from ._inference import np_fdr_bh, pvalues, sm_multipletests


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = int(param['n'])
    p = pvalues(n, param.get('seed', 0))
    jp = jax.block_until_ready(jnp.asarray(p))
    ref = np_fdr_bh()(p)  # exact numpy BH (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jp,) if framework == 'jax' else (p,)

    baselines = {
        'nitrix-jax': ('jax', lambda pv: fdr_bh(pv)[1]),  # p_adjusted
        'statsmodels.multipletests': ('statsmodels',
                                      sm_multipletests('fdr_bh')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_NS = [10_000, 100_000, 500_000]
_LARGE = [2_000_000, 8_000_000]

CASE = Case(
    name='fdr_bh',
    op_qualname='nitrix.stats.inference.fdr_bh',
    output_independent=False,  # the rank-scale + cummin couples the vector
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _NS],
    representative={'n': 100_000, 'seed': 0},
    large_param_points=tuple({'n': n, 'seed': 0} for n in _LARGE),
    complexity=(
        'an argsort (O(n log n)) + a rank-scaling + a reverse cumulative-min, '
        'over the n p-values. nitrix runs the whole vector on-device behind '
        'one compile; statsmodels sorts on the host. HBM ~ n. Scale axis = n, '
        'the whole-brain voxel count.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
