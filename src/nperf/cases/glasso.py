# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.glasso`` vs scikit-learn.

Graphical LASSO -- L1-penalised sparse precision (inverse covariance): given an
empirical covariance ``S[p, p]`` and penalty ``lam``, solve
``argmin_Theta  -logdet Theta + tr(S Theta) + lam |Theta|_1`` by block
coordinate descent.  **scikit-learn's ``graphical_lasso`` is the reference
implementation** (community baseline + fp64 oracle): both are iterative, so
they agree to convergence tolerance (~4e-4 on the precision, same support) --
gated at a loose tol, with the SAME ``S`` fed to both so the problem is
identical.  No on-device twin -> GPU headline nitrix-jax vs the sklearn CPU
bar.  The cost driver is ``p`` (O(p^3) per sweep); keyed ``c`` = ``p``.  Ratio
vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats import glasso

from ._base import BuiltPoint, Case
from ._shrinkage import sk_glasso, sparse_precision_cov

_LAM = 0.1  # L1 penalty (a moderately sparse precision)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = int(param['c'])
    S = sparse_precision_cov(p, param.get('seed', 0))  # well-conditioned
    sj = jax.block_until_ready(jnp.asarray(S))
    ref = sk_glasso(_LAM, fp64=True)(S)  # sklearn precision (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (sj,) if framework == 'jax' else (S,)

    baselines = {
        'nitrix-jax': ('jax', lambda s: glasso(s, _LAM)),
        # sklearn's graphical_lasso is a fp64 CPU tool -- its solver raises
        # "Non SPD" in fp32, so the baseline (like the oracle) runs fp64.
        'sklearn.graphical_lasso': ('numpy', sk_glasso(_LAM, fp64=True)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
        fidelity_note='both glasso impls are iterative block-coordinate '
                      'descent -> they agree to convergence tol (~4e-4 on the '
                      'precision; same support). The SAME empirical S is fed '
                      'to both, so the problem is identical; gated loosely.',
    )


# p = precision dimension (O(p^3) per coordinate sweep x n_outer x n_inner):
# kept modest -- glasso is the iterative, compute-heavy member of the family.
_SIZES = [40, 80, 160]
_LARGE = [320, 640]

CASE = Case(
    name='glasso',
    op_qualname='nitrix.stats.glasso',
    output_independent=False,  # a global L1-penalised inverse-covariance fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'seed': 0} for p in _SIZES],
    representative={'c': 80, 'seed': 0},
    large_param_points=tuple({'c': p, 'seed': 0} for p in _LARGE),
    complexity=(
        'block coordinate descent: O(p^3) per sweep x n_outer(100) x '
        'n_inner(50) -- the iterative, compute-heaviest member of the family '
        '(nitrix runs a fixed sweep count, sklearn early-exits on duality '
        'gap; a fixed-vs-early-stop wall-clock read). HBM ~ p^2. **FINDING: '
        'nitrix glasso is GPU-COMPILE-HOSTILE** -- the 100x50 sweep loop is '
        'UNROLLED (not lax.scan-rolled like the v4 registration recipes), so '
        'XLA builds a giant graph: GPU compile is infeasible even at p=32 '
        '(>5 min), while CPU runs fine (~0.3-60 s, p=40->320). So the GPU '
        'column times out (the documented finding) and the meaningful read is '
        'nitrix-CPU vs sklearn-CPU. FR: roll the glasso sweep loop (the '
        'registration loop-roll, applied to glasso).'),
    build=_build,
    rtol=1e-2,
    atol=2e-3,
)
