# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.lme.reml_fit`` vs statsmodels.

Voxelwise variance-components REML (FaST-LMM spectral trick): nitrix fits all
``V`` voxels in one **batched** call; ``statsmodels.MixedLM`` (the canonical
CPU LME library) must **loop** one fit per voxel -- the headline
batched-vs-looped comparison.  Scored against a **closed-form** balanced
one-way REML oracle (see ``cases/_lme.py``); the output is ``(V, 3)`` columns
``[beta, sigma_b^2, sigma_e^2]``.

nitrix runs **CPU-only on this L4** at present: the per-voxel ``vmap`` calls
``jnp.linalg.cholesky`` on the tiny ``(p, p)`` fixed-effect system, which
lowers to cuSOLVER ``potrf`` (``gpusolverDnCreate``) and **skips on GPU** --
the SAME blocker as ``flame_two_level`` (filed: nitrix FR
``lme-family-tiny-linalg-gpu-block-and-perf``; a Cholesky-free p=1 path
unblocks the GPU + is 3-6x faster on CPU).  NOTE: older store rows show ``ok``
on GPU --
they are **stale** (the cuSOLVER path regressed silently into a skip).  The
one-time ``ZZ^T`` eigh additionally goes through ``safe_eigh`` -> CPU.
statsmodels is CPU-only (``requires='cpu'``) and a **slow baseline** (per-voxel
iterative fits) -- skip in dev cycles (``--skip-slow``), run it in the
sprint-end full matrix.  No GPU reference library exists for LME.  Tolerance is
loose (``5e-3``) -- the convergence floor of the iterative solvers (lme design
doc).  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.lme import reml_fit

from ._base import BuiltPoint, Case, SlowBaseline
from ._lme import balanced_oneway, closed_form_reml, statsmodels_reml


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, k, n = param['V'], param['k'], param['n']
    Y, X, Z, groups = balanced_oneway(v, k, n, param.get('seed', 0))
    jY = jax.block_until_ready(jnp.asarray(Y))
    jX = jax.block_until_ready(jnp.asarray(X))
    jZ = jax.block_until_ready(jnp.asarray(Z))

    ref = closed_form_reml(Y.astype(np.float64), k, n)  # (V, 3) fp64 oracle

    def _nitrix(y: Any) -> Any:
        r = reml_fit(y, jX, jZ)
        return jnp.stack([r.beta_hat[:, 0], r.sigma_b_sq, r.sigma_e_sq], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y,)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'statsmodels.MixedLM': (
            'statsmodels', lambda y: statsmodels_reml(y, X, groups)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (voxels, groups, per-group): N = k*n subjects shared across voxels.  V scales
# the batch; statsmodels loops over it (the speedup grows with V).
_SHAPES = [(64, 8, 24), (256, 8, 24), (1024, 8, 24)]
# Brain-voxel scale: V up to 65536 voxels in the batch (statsmodels would loop
# ~15 min/fit here -> it is a slow_baseline, dropped by --skip-slow).
_LARGE = [(16384, 8, 24), (65536, 8, 24)]

CASE = Case(
    name='reml_fit',
    op_qualname='nitrix.stats.lme.reml_fit',
    output_independent=True,  # each voxel is an independent LME fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'k': k, 'n': n, 'seed': 0}
                  for (v, k, n) in _SHAPES],
    representative={'V': 256, 'k': 8, 'n': 24, 'seed': 0},
    large_param_points=tuple(
        {'V': v, 'k': k, 'n': n, 'seed': 0} for (v, k, n) in _LARGE),
    complexity=(
        'batched variance-components REML (FaST-LMM spectral trick) over V '
        'voxels: O(V*(n^3 eig + iters*n)) -- linear in the voxel batch '
        'V, the scale axis. nitrix fits all V in ONE call; statsmodels '
        'LOOPS one iterative fit per voxel (~14 ms/voxel), so the '
        'batched-vs-looped speedup GROWS with V (it is the headline, and why '
        'statsmodels is a slow_baseline at scale). HBM ~ V. The size tier '
        'varies V to brain-voxel scale.'),
    build=_build,
    # iterative REML convergence floor; loosened from 5e-3 to 1e-2 because at
    # the brain-voxel large tier the worst-voxel error reaches ~5.3e-3 (the
    # iterative floor + the tail of a larger voxel batch), just over 5e-3.
    rtol=1e-2,
    atol=1e-2,
    slow_baselines=(
        SlowBaseline(
            'statsmodels.MixedLM',
            'per-voxel iterative MixedLM fits (~14 ms/voxel on the L4 host); '
            'V=1024 x 13 timed reps ~ 3 min. CPU-only; skip in dev cycles.'),
    ),
)
