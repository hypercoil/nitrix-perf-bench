# -*- coding: utf-8 -*-
"""Tier-2 stats inference: ``nitrix.stats.inference.gpd_pvalue`` vs scipy.

Tail-accelerated permutation FWE p-value: fit a generalized Pareto distribution
to the top ``n_exceedances`` of the max-statistic null and read the tail
survival fraction, so the p-value is *continuous* in the extreme tail instead
of the discrete empirical ``#(null >= T) / n_perm`` (the Knijnenburg 2009 / FSL
``randomise`` GPD acceleration). The gate is the exact **numpy** reimpl of
nitrix's algorithm (``_inference.np_gpd``: method-of-moments GPD above the
(k+1)-th-largest threshold + the empirical body below it). **scipy**
``stats.genpareto`` (MLE fit + ``sf``) is a community cross-check (an
ApproxBaseline, not the oracle): it shares the empirical body bit-exactly but
its MLE fit differs from nitrix's MoM, so the GPD tail diverges up to ~10-15%
at the most extreme statistics. Keyed ``n`` = the number of observed
statistics evaluated (the voxel scale); the null is a fixed permutation-maxima
distribution. Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.inference import gpd_pvalue

from ._base import ApproxBaseline, BuiltPoint, Case
from ._inference import np_gpd, null_dist, scipy_gpd

_N_PERM = 2000  # fixed permutation-maxima null size


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = int(param['n'])
    seed = param.get('seed', 0)
    stat = null_dist(n, seed + 1)          # observed voxel statistics
    null = null_dist(_N_PERM, seed)        # the permutation-maxima null
    jstat = jax.block_until_ready(jnp.asarray(stat))
    jnull = jax.block_until_ready(jnp.asarray(null))
    # exact numpy reimpl of nitrix's MoM-GPD + empirical body (fp64) = oracle.
    ref = np_gpd(250)(stat, null)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jstat, jnull) if framework == 'jax' else (stat, null)

    baselines = {
        'nitrix-jax': ('jax', lambda s, nd: gpd_pvalue(s, nd)),
        # scipy genpareto-MLE: a DIFFERENT-fit community cross-check (nitrix is
        # method-of-moments) -- agrees in the empirical body, ~10-15% in the
        # GPD tail, so its fidelity is reported, not gated (ApproxBaseline).
        'scipy.genpareto': ('scipy', scipy_gpd(250)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_NS = [10_000, 100_000, 500_000]
_LARGE = [2_000_000]

CASE = Case(
    name='gpd_pvalue',
    op_qualname='nitrix.stats.inference.gpd_pvalue',
    output_independent=False,  # the shared GPD fit on the null couples it
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _NS],
    representative={'n': 100_000, 'seed': 0},
    large_param_points=tuple({'n': n, 'seed': 0} for n in _LARGE),
    complexity=(
        'fit a GPD to the top n_exceedances (=250) of the null, then evaluate '
        'the tail survival for each of the n observed statistics: a '
        'fixed-cost fit + O(n) eval (+ the null sort). nitrix runs it on-'
        'device behind one compile; scipy fits + evaluates on the host. HBM ~ '
        'n. Scale axis = n (observed-statistic / voxel count).'),
    build=_build,
    rtol=2e-3,  # vs the exact numpy MoM-GPD oracle (fp32 nitrix vs fp64)
    atol=1e-4,
    approximate_baselines=(
        ApproxBaseline(
            'scipy.genpareto',
            'scipy genpareto MLE (vs nitrix method-of-moments): matches the '
            'empirical body bit-exactly but the GPD tail diverges up to '
            '~10-15% at the most extreme statistics (fit-method difference, '
            'not a bug) -- reported, not gated. The neuro community GPD '
            'acceleration lives inside FSL randomise (no standalone tool), in '
            'permutation_test; scipy is a generic-stats cross-check.'),
    ),
)
