# -*- coding: utf-8 -*-
"""Tier-2 LME: ``nitrix.stats.lme.reml_fit(low_rank=True)`` -- the FaST-LMM
low-rank (q-rank) AI-REML, scaled on ``N``.

When the random-effect design ``Z`` is low-rank (``q = rank(Z) << N`` -- a
handful of group/grouping columns), the FaST-LMM **low-rank** diagonalisation
fits the two-component REML in ``O(N q^2)`` per voxel instead of the full
``O(N^3)``: one ``ZZ^T`` eigendecomposition gives the ``q`` nonzero
eigenvalues, and the AI-REML iterates in that diagonal basis.  So unlike the
full-rank ``reml_fit`` case (which scales the voxel batch ``V``), the headline
here is the **observation count ``N``**: nitrix's low-rank fit stays cheap as
``N`` grows, while the community tools -- R ``lme4 lmer`` and
``statsmodels.MixedLM``, looped one iterative fit per voxel -- pay the full
per-fit cost and become infeasible (minutes->hours, dropped as slow).

Fidelity: the closed-form balanced-one-way REML oracle (exact for any ``N``).
**The low-rank path is numerically EXACT** -- in float64 it matches the oracle
(and the full-rank path) to ~0.  The loose gate reflects an **fp32 precision
floor**, not the algorithm: the ``ZZ^T`` eigendecomposition (``safe_eigh`` ->
CPU) loses digits as the eigenvalue spread widens with ``N`` (verified: rel-to-
tol ~0.07 at N=192 -> ~0.33 by N>=1600, flat across n_iter -- conditioning, not
convergence).  This is the same fp32-at-scale character as the Lomb-Scargle
periodogram; nitrix FR ``lme-lowrank-eigh-fp32-conditioning`` notes that the
low-rank ``ZZ^T`` eigh is more fp32-sensitive than the full-rank path.  Keyed
``{V, q, N}`` (scale axis = ``N``; ``V``/``q`` fixed; the large tier stops at
N=16384, beyond which the fp32 floor exceeds the gate).  Ratio vs
``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.lme import reml_fit

from ._base import ApproxBaseline, BuiltPoint, Case, SlowBaseline
from ._lme import (
    balanced_oneway,
    closed_form_reml,
    r_lme4_iofloor,
    r_lme4_reml,
    statsmodels_reml,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, q, big_n = int(param['V']), int(param['q']), int(param['N'])
    n = big_n // q  # per-group count (N = q * n, balanced)
    Y, X, Z, groups = balanced_oneway(v, q, n, param.get('seed', 0))
    ref = closed_form_reml(Y.astype(np.float64), q, n)  # (V,3) exact oracle
    jY = jax.block_until_ready(jnp.asarray(Y))
    jX = jax.block_until_ready(jnp.asarray(X))
    jZ = jax.block_until_ready(jnp.asarray(Z))

    def _nitrix(y: Any) -> Any:  # the LOW-RANK FaST-LMM path
        r = reml_fit(y, jX, jZ, low_rank=True)
        return jnp.stack([r.beta_hat[:, 0], r.sigma_b_sq, r.sigma_e_sq], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y,)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'statsmodels.MixedLM': (
            'statsmodels', lambda y: statsmodels_reml(y, X, groups)),
        'R.lme4': ('r', lambda y: r_lme4_reml(y, groups)),
        'R.iofloor': ('r', lambda y: r_lme4_iofloor(y, groups)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
        fidelity_note=(
            'low_rank=True is numerically EXACT (matches the oracle to ~0 in '
            'float64); the loose gate is the fp32 ZZ^T-eigh precision floor '
            'that grows with N (conditioning, not convergence -- FR '
            'lme-lowrank-eigh-fp32-conditioning).'),
    )


# (V voxels, q = rank(Z), N observations): N is the scale axis (the FaST-LMM
# low-rank win); V and q fixed.  Representative kept small so R lme4 /
# statsmodels (looped) stay tractable + drift is fast.
_DEV = [(128, 8, 800), (128, 8, 1600), (128, 8, 3200)]
# Large tier: N to where the looped community fits are infeasible (dropped via
# --skip-slow); nitrix's low-rank fit stays cheap (the headline). V modest so
# the oracle + the device batch fit comfortably.
_LARGE = [(128, 8, 8192), (128, 8, 16384)]


def _p(v: int, q: int, big_n: int) -> Dict[str, Any]:
    return {'V': v, 'q': q, 'N': big_n, 'low_rank': True, 'seed': 0}


CASE = Case(
    name='reml_fit_lowrank',
    # same op as reml_fit, a distinct case (the low-rank path / N-scaling); the
    # drift manifest is keyed by case name, so >1 case may target one op.
    op_qualname='nitrix.stats.lme.reml_fit',
    output_independent=True,  # each voxel is an independent LME fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[_p(*p) for p in _DEV],
    representative=_p(128, 8, 1600),
    large_param_points=tuple(_p(*p) for p in _LARGE),
    complexity=(
        'FaST-LMM low-rank (q-rank) AI-REML: one ZZ^T eigendecomposition (q '
        'nonzero eigenvalues) + iterate in the diagonal basis -> O(N q^2) per '
        'voxel, vs the full O(N^3). nitrix batches all V voxels on-device and '
        'stays cheap as N grows; R lme4 / statsmodels LOOP one full iterative '
        'fit per voxel, so they become infeasible at large N (the headline, '
        'and why both are slow_baselines). Scale axis = N (observations).'),
    build=_build,
    rtol=1e-2,  # iterative REML + the fp32 ZZ^T-eigh floor (exact in fp64)
    atol=1e-2,
    slow_baselines=(
        SlowBaseline(
            'statsmodels.MixedLM',
            'per-voxel iterative MixedLM fits; the per-fit cost grows with N, '
            'so it is infeasible at the large-N tier. CPU-only; skip in dev.'),
        SlowBaseline(
            'R.lme4',
            'per-voxel R lme4 lmer fits looped in one Rscript; per-fit cost '
            'grows with N -> infeasible at the large-N tier. CPU-only; skip.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'R.iofloor',
            'no-op: times the CSV round-trip + R startup R.lme4 pays '
            '(economic subtracts the same-namespace floor); not scored.'),
    ),
)
