# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.morphology.median_filter`` vs scipy.

The **no-cross-impl-oracle** case.  Both compute a true windowed median, but
with *different boundary policies*: nitrix pads with NaN and medians over the
available in-bounds subset (a shrinking window at the border), while
``scipy.ndimage.median_filter`` medians a full window via a boundary mode
(default ``reflect``).  Interiors match exactly; the borders differ **by
design**, so there is no single oracle both should match.

So the case sets ``fp64_reference=None``: the attempt is still OK, the perf
ratio is still a fair task-level comparison (both denoise with a size-3
median), and the numerical check is recorded as *inconclusive* with the reason
below (the suite refuses to manufacture a fake oracle).  Ratio via
``--reference scipy.ndimage.median_filter``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.morphology import median_filter

from ._base import BuiltPoint, Case, to_cupy


def _cupy_median(x: Any, size: int) -> Any:
    '''GPU median_filter (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).
    Reflect boundary like scipy -- so, like scipy, it differs from nitrix's
    NaN-pad shrink-window at borders; this case has no oracle, perf only.'''
    from cupyx.scipy import ndimage as cnd

    return cnd.median_filter(x, size=size)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    size = param.get('size', 3)
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal(shape).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: median_filter(x, size=size)),
        'scipy.ndimage.median_filter': (
            'scipy', lambda x: spnd.median_filter(x, size=size)),
        'cupyx.scipy.ndimage.median_filter': (
            'cupy', lambda x: _cupy_median(x, size)),  # GPU ref (perf only)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,  # no shared oracle (boundary policies differ)
        ratio_reference='nitrix-jax',
        fidelity_note='boundary divergence: nitrix NaN-pad shrink-window '
                      'median vs scipy reflect; interiors match, perf is '
                      'task-comparable',
    )


_SHAPES = [[64, 64], [256, 256]]

CASE = Case(
    name='median_filter',
    op_qualname='nitrix.morphology.median_filter',
    output_independent=False,  # each output is a median over a size-window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'size': 3, 'seed': 0} for s in _SHAPES],
    representative={'shape': [256, 256], 'size': 3, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
