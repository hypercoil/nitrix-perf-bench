# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.stats.corr`` vs ``numpy.corrcoef``.

Sibling of the ``cov`` case (see its docstring): nitrix op on jax vs the numpy
reference, both scored against an fp64 oracle, ratio read via
``--reference numpy.corrcoef``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import corr

from ._base import BuiltPoint, Case, to_cupy


def _cupy_corrcoef(x: Any) -> Any:
    '''CuPy ``corrcoef`` (GPU); cupy imported lazily (refs-cupy env only).'''
    import cupy as cp

    return cp.corrcoef(x)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, t = param['n'], param['t']
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal((n, t)).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = np.corrcoef(X.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: corr(x)),
        'numpy.corrcoef': ('numpy', lambda x: np.corrcoef(x)),
        'cupy.corrcoef': ('cupy', _cupy_corrcoef),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (variables, samples): same ladder as cov.
_SHAPES = [(50, 500), (500, 2000), (2000, 1000)]
_LARGE = [(4000, 2000), (8000, 2000)]  # large parcellation

CASE = Case(
    name='corr',
    op_qualname='nitrix.stats.corr',
    output_independent=False,  # corr[i, j] couples rows i and j (annex §C)
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 't': t, 'seed': 0} for (n, t) in _SHAPES],
    representative={'n': 2000, 't': 1000, 'seed': 0},
    large_param_points=tuple(
        {'n': n, 't': t, 'seed': 0} for (n, t) in _LARGE),
    complexity=(
        'corrcoef = the centred/standardised cov: O(n^2 * t) -- a single '
        'BLAS-class matmul (same GPU-friendly regime as cov), the larger n '
        'being where it pulls ahead. HBM ~ n^2 (the n x n output). The size '
        'tier varies n to large-parcellation scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
