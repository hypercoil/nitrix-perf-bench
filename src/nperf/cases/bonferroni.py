# -*- coding: utf-8 -*-
"""Tier-2 inference: ``nitrix.stats.inference.bonferroni`` vs statsmodels.

Bonferroni family-wise correction of ``n`` p-values: ``p_adjusted =
min(p * n, 1)``. The community tool is **statsmodels**
``multipletests(method='bonferroni')`` (exact); the fp64 oracle is the
elementwise **numpy** ``clip(p * n, 0, 1)``. A trivially-parallel elementwise
map -- the perf story is purely device throughput over the whole-brain vector
(no host sort, unlike ``fdr_bh``). ``bonferroni`` returns ``(rejected,
p_adjusted)``; the case scores ``p_adjusted``. Keyed ``n`` = the number of
p-values (scale axis). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.inference import bonferroni

from ._base import BuiltPoint, Case
from ._inference import np_bonferroni, pvalues, sm_multipletests


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = int(param['n'])
    p = pvalues(n, param.get('seed', 0))
    jp = jax.block_until_ready(jnp.asarray(p))
    ref = np_bonferroni()(p)  # exact numpy Bonferroni (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jp,) if framework == 'jax' else (p,)

    baselines = {
        'nitrix-jax': ('jax', lambda pv: bonferroni(pv)[1]),  # p_adjusted
        'statsmodels.multipletests': ('statsmodels',
                                      sm_multipletests('bonferroni')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_NS = [10_000, 100_000, 500_000]
_LARGE = [2_000_000, 8_000_000]

CASE = Case(
    name='bonferroni',
    op_qualname='nitrix.stats.inference.bonferroni',
    output_independent=True,  # elementwise min(p*n, 1) -- each entry is local
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _NS],
    representative={'n': 100_000, 'seed': 0},
    large_param_points=tuple({'n': n, 'seed': 0} for n in _LARGE),
    complexity=(
        'an elementwise min(p * n, 1) over the n p-values: O(n), '
        'embarrassingly parallel (no sort). nitrix maps the whole vector '
        'on-device behind one compile. HBM ~ n. Scale axis = n (the '
        'whole-brain voxel count).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
