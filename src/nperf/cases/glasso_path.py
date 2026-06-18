# -*- coding: utf-8 -*-
"""Tier-2 stats: ``nitrix.stats.glasso_path`` vs scikit-learn.

Warm-started graphical-LASSO **regularisation path**: solve glasso at each of
``L`` penalties ``lambdas`` (warm-starting each from the previous, denser
solution) -> ``precision[L, p, p]``.  The model-selection workhorse (pick the
path point minimising EBIC).  **scikit-learn** has no batched path, so the
oracle/baseline is ``graphical_lasso`` looped over the same ``lambdas`` in fp64
(the SAME ``S`` fed to both).  nitrix vmaps/scans the warm-started path on the
GPU; the community bar is the CPU loop -- the batching story (gap grows with
``L``).  Keyed ``c`` = ``p``; ``L`` fixed.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats import glasso_path

from ._base import BuiltPoint, Case
from ._shrinkage import sk_glasso_path, sparse_precision_cov

# L1 penalties (coarse->fine, dense->sparse): the warm-started path. Floored at
# 0.05 -- a smaller penalty drives the precision toward singular, where sklearn
# raises "Non SPD" (nitrix's fixed-sweep solver tolerates it; we keep the
# comparison in the regime BOTH solve).
_LAMBDAS = np.geomspace(0.3, 0.05, 8).astype(np.float32)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    p = int(param['c'])
    S = sparse_precision_cov(p, param.get('seed', 0))
    sj = jax.block_until_ready(jnp.asarray(S))
    lam_j = jnp.asarray(_LAMBDAS)
    ref = sk_glasso_path(_LAMBDAS, fp64=True)(S)  # sklearn loop fp64 = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (sj, lam_j) if framework == 'jax' else (S,)

    baselines = {
        'nitrix-jax': ('jax', lambda s, la: glasso_path(s, la)),
        # fp64: sklearn's graphical_lasso solver raises "Non SPD" in fp32.
        'sklearn.graphical_lasso_path': (
            'numpy', sk_glasso_path(_LAMBDAS, fp64=True)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
        fidelity_note='iterative glasso at each lambda -> agreement to '
                      'convergence tol; SAME S + lambdas to both. nitrix '
                      'warm-starts the path on-device while the sklearn loop '
                      'solves each lambda FRESH, so at the smallest (densest) '
                      'lambda the two diverge a touch more (~1.5% on large '
                      'precision entries) -- gated loosely. Also the '
                      'batching/amortisation comparison (one compile vs L '
                      'CPU solves).',
    )


_SIZES = [40, 80, 160]
_LARGE = [320, 640]

CASE = Case(
    name='glasso_path',
    op_qualname='nitrix.stats.glasso_path',
    output_independent=False,  # warm-started: each lambda couples to the prev
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'c': p, 'seed': 0} for p in _SIZES],
    representative={'c': 80, 'seed': 0},
    large_param_points=tuple({'c': p, 'seed': 0} for p in _LARGE),
    complexity=(
        f'L={len(_LAMBDAS)} warm-started glasso solves (each O(p^3) x sweep). '
        'nitrix runs the path on-device (one compile, warm-started scan); '
        'sklearn loops L CPU solves -> the gap grows with L (the path '
        'batching story). HBM ~ L*p^2 (the stacked precisions). The size tier '
        'grows p; sklearn looped is the CPU bar. FINDING: like glasso, the '
        'UNROLLED sweep loop (x L lambdas) makes nitrix GPU-compile-hostile '
        '-> the GPU column times out; nitrix-CPU vs the sklearn loop is the '
        'meaningful read (FR: roll the sweep loop).'),
    build=_build,
    rtol=2e-2,   # iterative + warm-start-vs-fresh divergence at small lambda
    atol=5e-3,
)
