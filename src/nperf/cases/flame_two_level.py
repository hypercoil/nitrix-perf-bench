# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.lme.flame_two_level`` vs closed form.

Voxelwise FLAME-style two-level group model (FSL FLAME equivalent): a
single-parameter REML for the between-subject variance ``sigma_b^2`` given
**known** per-subject within-variance, batched over all ``V`` voxels.

**Reference.**  There is no fair external perf competitor: the only Python LME
library (``statsmodels``) cannot consume known per-subject variances, and FSL
FLAME (the upstream tool) is not importable -- it is the *fair* competitor, to
be revisited in the external-tool workstream.  So this case reports nitrix's
**CPU-vs-GPU** behaviour (does the batched fit scale on the device?) and scores
fidelity against a **closed-form** oracle.  The closed form is exact only for
the **constant** within-variance case (then the model covariance is
``(sigma_b^2 + s2) I`` and the REML collapses to OLS + residual-variance; see
``cases/_lme.py``) -- verified to match ``flame_two_level`` to ~2e-4.  It is
the oracle, **not** a perf baseline (timing the general Newton solver against a
degenerate-case closed form would be apples-to-oranges), so the only baseline
is ``nitrix-jax`` itself.  Output ``(V, 2)`` = ``[gamma, sigma_b^2]``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.lme import flame_two_level

from ._base import BuiltPoint, Case
from ._lme import flame_closed_form, flame_input

_S2 = 0.3  # constant known within-variance (enables the closed-form oracle)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, big_n = param['V'], param['N']
    beta, varw, x_group = flame_input(v, big_n, param.get('seed', 0), s2=_S2)
    jb = jax.block_until_ready(jnp.asarray(beta))
    jv = jax.block_until_ready(jnp.asarray(varw))
    jx = jax.block_until_ready(jnp.asarray(x_group))

    ref = flame_closed_form(beta.astype(np.float64),
                            x_group.astype(np.float64), _S2)  # (V, 2) oracle

    def _nitrix(b: Any) -> Any:
        r = flame_two_level(b, jv, jx)
        return jnp.stack([r.gamma_hat[:, 0], r.sigma_b_sq], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jb,) if framework == 'jax' else (beta,)

    # Only nitrix-jax: no fair external competitor (see module docstring).  The
    # ratio is within-platform vs nitrix-jax (1.0); the value is the CPU-vs-GPU
    # cross-platform comparison plus fidelity vs the closed-form oracle.
    baselines = {'nitrix-jax': ('jax', _nitrix)}
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (voxels, subjects): N subjects per voxel (typical fMRI group size); V scales
# the batch.  Larger V is where the device-resident batched fit pulls ahead.
_SHAPES = [(1024, 60), (8192, 60), (65536, 60)]
# Brain-volume scale; on GPU these hit the cuSOLVER ceiling (see complexity) so
# they record a skip there and run on CPU -- the documented scale ceiling.
_LARGE = [(131072, 60), (262144, 60)]

CASE = Case(
    name='flame_two_level',
    op_qualname='nitrix.stats.lme.flame_two_level',
    output_independent=True,  # each voxel is an independent FLAME fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'N': n, 'seed': 0} for (v, n) in _SHAPES],
    representative={'V': 8192, 'N': 60, 'seed': 0},
    large_param_points=tuple(
        {'V': v, 'N': n, 'seed': 0} for (v, n) in _LARGE),
    complexity=(
        'batched single-param REML for the between-subject variance over V '
        'voxels (FSL FLAME equiv): O(V * iters * N) -- linear in the voxel '
        'batch V. MEASURED (L4): scales cleanly on the GPU through the dev '
        'tier to V=65536, but V>=131072 fails the GPU SOLVER (gpusolverDnCreate '
        '-- a cuSOLVER-class blocker on this box), a hard GPU scale CEILING; '
        'the batched FLAME still runs on CPU at brain-volume V. No '
        'fair external perf competitor (statsmodels cannot consume known '
        'per-subject variances; FSL FLAME is file-coupled). HBM ~ V.'),
    build=_build,
    rtol=5e-3,  # iterative-solver convergence floor (lme design doc)
    atol=5e-3,
)
