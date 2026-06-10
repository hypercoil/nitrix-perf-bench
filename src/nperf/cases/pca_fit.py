# -*- coding: utf-8 -*-
"""Tier-2 (stats breadth): ``nitrix.stats.pca_fit`` vs scikit-learn / cupy.

The eigh-bound *fit*: the default ``solver='full'`` forms the ``(d, d)``
covariance and eigendecomposes it.  System-under-test is the nitrix op on jax;
the CPU task-level floor is ``sklearn.decomposition.PCA`` (exact full SVD), and
the GPU on-target twin is a CuPy covariance-eigh.  Scored against an fp64
oracle on the **explained_variance** (the top-``k`` covariance eigenvalues) --
the sign/rotation-invariant fidelity quantity (principal axes carry a +/- and a
within-eigenspace rotation ambiguity that makes a direct component comparison
ill-posed; the eigenvalues are unique).  Basis-application correctness is
covered by the ``pca_transform`` / ``pca_inverse_transform`` cases.

GPU note (re-measured 2026-06; see [[perfbench-gpu-eigh-blocker]]): the older
"safe_eigh routes the eigh to CPU at d>=256" assumption did **not** reproduce
here -- in a *fresh* worker the cuSOLVER eigh handle initialises and the
covariance eigh runs **GPU-native through d=2048** (nitrix 44.9 ms vs the CuPy
device-eigh 41.6 ms at d=2048: parity, ~0.93x, no host round-trip).
``safe_eigh``'s CPU fallback is a latent safety net that did not fire in these
measurements; cuSOLVER handle-creation failures appear only in long-lived /
memory-pressured contexts (observed in a reused REPL, not in the per-attempt
workers).  So the honest result is **GPU parity with cupy** + a **CPU win over
sklearn** (nitrix's eigh-of-cov is structurally cheaper than sklearn's full SVD
of the data; 6-12x on CPU).  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import pca_fit

from ._base import BuiltPoint, Case, SlowBaseline, to_cupy
from ._pca import (
    cupy_explained_variance,
    np_explained_variance,
    pca_input,
    sklearn_explained_variance,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n, d, k = param['n'], param['d'], param['k']
    X = pca_input(n, d, k, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))

    ref = np_explained_variance(X.astype(np.float64), k)  # (k,) fp64 oracle

    def _nitrix(x: Any) -> Any:
        return pca_fit(x, n_components=k).explained_variance

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'sklearn.PCA': ('sklearn', sklearn_explained_variance(k)),  # CPU floor
        'cupy.eigh_cov': ('cupy', cupy_explained_variance(k)),  # GPU twin
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (samples, features, components): d spans the feature axis; the eigh is on the
# (d, d) cov. Measured GPU-native (cuSOLVER) through the whole range here.
_SHAPES = [(2048, 128, 16), (2048, 256, 16), (1024, 512, 16)]
# Brain-feature scale: d up to ~2048 features (a parcellation as the feature
# axis). nitrix's GPU eigh stayed GPU-native to d=2048 (~45 ms, parity with the
# cupy device-eigh); sklearn's full SVD of (8192, 2048) is the long pole (it
# times out -> sklearn.PCA is a slow_baseline).
_LARGE = [(4096, 1024, 32), (8192, 2048, 32)]

CASE = Case(
    name='pca_fit',
    op_qualname='nitrix.stats.pca_fit',
    # the eigendecomposition couples every covariance entry -- not element-wise
    # independent; the (k,) eigenvalue oracle is computed in full (cheap).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'd': d, 'k': k, 'seed': 0}
                  for (n, d, k) in _SHAPES],
    representative={'n': 2048, 'd': 256, 'k': 16, 'seed': 0},
    large_param_points=tuple(
        {'n': n, 'd': d, 'k': k, 'seed': 0} for (n, d, k) in _LARGE),
    complexity=(
        'cov is O(n * d^2) (one BLAS matmul); the eigh of the (d, d) cov is '
        'O(d^3) and dominates at brain-feature d. HBM ~ d^2 (the cov). '
        'MEASURED (L4): the cuSOLVER eigh stayed GPU-native through d=2048 in '
        'fresh workers (NO CPU fallback fired; the older d>=256 routing did '
        'not reproduce), so nitrix is at PARITY with the cupy device-eigh on '
        'GPU (~0.93-0.96x; cupy marginally faster, 0.63x at tiny d=128 where '
        'nitrix fixed overhead dominates). The WIN is on CPU vs sklearn '
        '(6-12x): nitrix eigh-decomposes the (d, d) cov where sklearn SVDs '
        'the (n, d) data -- structurally cheaper when n>d. GPU-vs-CPU for '
        'nitrix is ~30x at d=2048. The size tier varies d to parcel scale.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    slow_baselines=(
        SlowBaseline(
            'sklearn.PCA',
            'exact full-SVD of the (n, d) data; the large tier (8192, 2048) '
            'x 13 reps exceeds the worker timeout on the CPU platform '
            '(~3 s/call but high variance; timed out at d=2048). CPU-only '
            'exact PCA; skip in dev cycles, run in the sprint-end matrix.'),
    ),
)
