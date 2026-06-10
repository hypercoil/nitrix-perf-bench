# -*- coding: utf-8 -*-
"""Morphology: ``nitrix.morphology.distance_transform_edt`` vs scipy / cupyx.

The exact Euclidean distance transform of a binary mask.  **nitrix's is a thin
alias for ``distance_transform(metric='euclidean')`` -- the same separable
min-plus-matmul SEMIRING engine, which searches over *all* parabolas rather
than the Felzenszwalb-Huttenlocher (F-H) lower-envelope.**  So the result
matches scipy exactly (an exact co-oracle, ~1.1e-7), but the cross-tool *speed*
comparison is **semiring-brute-force (nitrix) vs F-H (scipy / cupyx
``distance_transform_edt``)** -- cupyx the on-target GPU bar.

This is the **depth-vs-FLOP crossover the EDT family originally motivated**
(COVERAGE_MANDATE §2.6; the ``distance_transform`` exemplar): the all-parabola
semiring is shallow + high-FLOP, so on a parallel GPU it WINS at small scale
(depth-bound) and LOSES at large scale (FLOP-bound).  This is the **deliberate,
known** semiring trade-off (the semiring was *chosen* because it beats F-H at
small scale), not a regression.  Global op, GPU-pure.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import distance_transform_edt

from ._base import BuiltPoint, Case, to_cupy
from ._connectivity import blob_mask, cupyx_edt, scipy_edt


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    mask = blob_mask((d, d, d), param.get('seed', 0))
    mj = jax.block_until_ready(jnp.asarray(mask))
    ref = scipy_edt(mask).astype('float64')  # fp64 co-oracle (== scipy)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (mj,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: distance_transform_edt(m)),
        'scipy.distance_transform_edt': ('scipy', scipy_edt),  # oracle/floor
        'cupy.distance_transform_edt': ('cupy', cupyx_edt()),  # GPU bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='distance_transform_edt',
    op_qualname='nitrix.morphology.distance_transform_edt',
    output_independent=False,  # a global transform (distance to the boundary)
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'separable min-plus SEMIRING EDT (all-parabola search, the euclidean '
        'alias of distance_transform): high-FLOP but shallow/parallel, so it '
        'WINS small and LOSES large vs F-H. MEASURED (L4): nitrix 2.4x ahead '
        'of cupyx at 48^3, crossing over ~96^3 to 2.2x behind at 160^3 '
        '(0.15->1.99 vs cupyx 0.36->0.89 ms) -- the known semiring trade-off. '
        'A scale-aware dispatch (semiring small, F-H large) keeps the win at '
        'both ends (filed lower-priority on nitrix main).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
