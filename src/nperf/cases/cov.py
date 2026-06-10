# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.stats.cov`` vs ``numpy.cov``.

The first of the external-library reference cases migrated out of nitrix's
in-tree ``bench/PERF_AUDIT.md`` (nitrix BACKLOG B11): the system-under-test is
the nitrix op on jax; the reference is the natural CPU library a practitioner
would otherwise reach for (here numpy).  Both are scored against an fp64 oracle
(``numpy.cov`` in double on the same values), so the fidelity gate measures
round-off, and the ratio (``--reference numpy.cov``) reads as "nitrix is N×
numpy" -- exactly the op_matrix's perf cell.

numpy is a core dep, so the CPU reference runs in the same worker env as the
jax baseline; no refs env is needed (unlike torch / PyG).  The GPU reference is
``cupy.cov`` -- CuPy's ``cov``, the on-device twin of ``numpy.cov`` -- which
gives the *apples-to-apples* GPU comparison the numpy/scipy CPU floor cannot
(Phase B / COVERAGE_MANDATE Thrust 3).  Its provider is GPU-only
(``requires='gpu'``): it runs in the isolated refs-cupy env and is recorded
``platform_not_applicable`` on the CPU platform.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import cov

from ._base import BuiltPoint, Case, to_cupy


def _cupy_cov(x: Any) -> Any:
    '''CuPy ``cov`` (GPU); cupy imported lazily so only the cupy worker (the
    refs-cupy env) needs it -- the jax / numpy workers never import it.'''
    import cupy as cp

    return cp.cov(x, bias=False)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    c, n_obs = param['c'], param['n_obs']
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal((c, n_obs)).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    # fp64 oracle: numpy.cov in double on the same values.
    ref = np.cov(X.astype(np.float64), bias=False)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: cov(x)),
        'numpy.cov': ('numpy', lambda x: np.cov(x, bias=False)),
        'cupy.cov': ('cupy', _cupy_cov),  # GPU on-target ref (requires gpu)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (channels, observations): the PERF_AUDIT ladder -- (50,500) is near parity,
# the larger shapes are where the BLAS path pulls ahead.
_SHAPES = [(50, 500), (500, 2000), (2000, 1000)]
_LARGE = [(4000, 2000), (8000, 2000)]  # large parcellation / sub-parcellation

CASE = Case(
    name='cov',
    op_qualname='nitrix.stats.cov',
    # cov[i, j] couples rows i and j over all observations -- not an
    # element-wise-independent output; the fp64 oracle is computed in full
    # (cheap), so this is documentary only (annex §C).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': c, 'n_obs': n, 'seed': 0} for (c, n) in _SHAPES],
    representative={'c': 2000, 'n_obs': 1000, 'seed': 0},
    large_param_points=tuple(
        {'c': c, 'n_obs': n, 'seed': 0} for (c, n) in _LARGE),
    complexity=(
        'centred X @ X.T / (n-1): O(c^2 * n_obs) -- a single BLAS-class '
        'matmul, the GPU-friendly regime (the larger c is where the matmul '
        'path pulls ahead of the CPU floor). HBM ~ c^2 (c x c output). The '
        'size tier varies c to large-parcellation scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
