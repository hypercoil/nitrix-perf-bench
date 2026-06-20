# -*- coding: utf-8 -*-
"""Marquee (v3): ``nitrix.stats.glmm.glmm_fit`` -- mass-univariate GLMM/PQL.

The v3 surface with the largest modelling-path space: per voxel,
``g(E[y|b]) = X beta + b[group]`` under a GLM family, dispatched on the
random-effect level count (few ``q<=64`` dense vs many ``q>64`` structured
Schur), with random **slopes** served by the robust joint-Schur + REML-EM
solver or adaptive Gauss-Hermite quadrature.  That combinatorial space
(family x structure x method x tier) is too large to probe densely at brain
scale, and the looped-CPU gold standard (R ``mgcv::gam`` re-smooth) is
infeasible there -- so we measure a dense range of **small, fast scales per
path** and *extrapolate* (``tools/extrapolate_report.py``), keeping the
numerically-hard cells v3 hardened (the robust unstructured-slope solver, AGQ
for binary clusters) measured too, plus one measured brain-scale anchor per
scaling story to validate the guess.

Paths (each a ``ScalePath`` with its own scale axis + theoretical exponent):

- ``gaussian-intercept-few`` (V-sweep) -- Gaussian random intercept *is* the
  LME, so the closed-form balanced REML is the fp64 oracle; the few-tier
  batched throughput story.  Anchor: whole-brain V.
- ``binomial-intercept-many`` (q-sweep, all q>64) -- the headline: nitrix's
  structured many-tier is O(q) while the dense looped reference is ~O(q^3),
  so the projected brain-scale speedup over R grows.  Anchor: q=1024
  per-subject random intercepts.
- ``poisson-intercept-few`` (V-sweep) -- low-count log-link counts; few-tier
  throughput, a second family.
- ``gaussian-slope-unstructured`` (V-sweep, **challenging**) -- the robust
  correlated-slope solver v3 hardened against indefinite Hessians; the
  recovery test pins finiteness (no divergence).
- ``binomial-slope-agq`` (V-sweep, **challenging**) -- adaptive Gauss-Hermite
  (the PQL-attenuation correction for binary slopes); the n_quad^r mode grid
  is a constant factor, V-scaling stays batched-linear.

No GPU GLMM reference library exists (as for LME): ratio vs ``nitrix-jax``,
with a GPU-vs-own-CPU economic verdict and the extrapolated speedup over the
(infeasible-at-scale) looped R baseline.  The non-Gaussian intercept paths
carry that looped R baseline (a ``slow_baseline``); the slope / AGQ paths are
nitrix-only (correctness = planted-truth recovery + finiteness).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.glmm import glmm_fit

from ._base import (
    ApproxBaseline,
    BuiltPoint,
    Case,
    CostLaw,
    ScalePath,
    SlowBaseline,
    scaling_sweep,
)
from ._glmm import glmm_data, r_mgcv_glmm, r_mgcv_iofloor
from ._lme import closed_form_reml


def _build(param: Dict[str, Any]) -> BuiltPoint:
    fam = param['family']
    structure = param['structure']            # intercept | unstructured | ...
    method = param.get('method', 'pql')
    v, q, n_per = int(param['V']), int(param['q']), int(param['n_per'])
    seed, n_quad = int(param.get('seed', 0)), int(param.get('n_quad', 5))

    Y, X, group, z, truth = glmm_data(fam, v, q, n_per, structure, seed)
    jY = jax.block_until_ready(jnp.asarray(Y))
    jX = jax.block_until_ready(jnp.asarray(X))
    jgroup = jax.block_until_ready(jnp.asarray(group))
    jz = jax.block_until_ready(jnp.asarray(z)) if z is not None else None
    is_intercept = structure == 'intercept'
    struct_arg = 'unstructured' if is_intercept else structure

    def _nitrix(y: Any) -> Any:
        r = glmm_fit(y, jX, group=jgroup, z=jz, structure=struct_arg,
                     family=fam, method=method, n_quad=n_quad)
        if not is_intercept:
            return r.beta_hat              # (V, 2) fixed effects
        if fam == 'gaussian':
            return jnp.stack([r.beta_hat[:, 0], r.re_var, r.dispersion], -1)
        return jnp.stack([r.beta_hat[:, 0], r.re_var], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y,)

    baselines = {'nitrix-jax': ('jax', _nitrix)}
    ref: Any = None
    note: Any = None
    if is_intercept and fam == 'gaussian':
        # Gaussian random intercept == LME -> closed-form balanced REML oracle.
        ref = closed_form_reml(Y.astype(np.float64), q, n_per)   # (V, 3)
    elif is_intercept:
        # Non-Gaussian: the looped R mgcv re-smooth is the gold-standard PQL.
        baselines['R.mgcv'] = ('r', lambda y: r_mgcv_glmm(y, group, fam))
        baselines['R.iofloor'] = ('r', lambda y: r_mgcv_iofloor(y, group))
        note = (f'{fam} random-intercept GLMM (PQL): no closed-form oracle; '
                'correctness = agreement with R mgcv s(g,bs="re") REML (the '
                'PQL estimator nitrix matches) + finiteness.')
    else:
        note = (f'{fam} random slope ({method}, {struct_arg}): the v3 '
                'hardened path (robust joint-Schur+REML-EM / AGQ). No closed '
                'form or wired external ref; correctness = the fixed effects '
                'recover the planted truth + finite variance comps (no '
                'divergence).')
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax', fidelity_note=note,
    )


# --- modelling paths (config space), each a dense small-fast scale sweep ---
_PATHS = [
    ScalePath(
        'gaussian-intercept-few',
        {'family': 'gaussian', 'structure': 'intercept', 'method': 'pql',
         'q': 8, 'n_per': 24, 'seed': 0},
        CostLaw('V', 1.0, regime='few-tier dense, batched over V'),
        (64, 128, 256, 512, 1024)),
    ScalePath(
        'binomial-intercept-many',
        {'family': 'binomial', 'structure': 'intercept', 'method': 'pql',
         'V': 64, 'n_per': 8, 'seed': 0},
        CostLaw('q', 1.0, regime='many-tier q>64 structured Schur'),
        (96, 128, 192, 256, 384)),
    ScalePath(
        'poisson-intercept-few',
        {'family': 'poisson', 'structure': 'intercept', 'method': 'pql',
         'q': 8, 'n_per': 24, 'seed': 0},
        CostLaw('V', 1.0, regime='few-tier dense, log-link counts'),
        (64, 128, 256, 512, 1024)),
    ScalePath(
        'gaussian-slope-unstructured',
        {'family': 'gaussian', 'structure': 'unstructured', 'method': 'pql',
         'q': 16, 'n_per': 24, 'seed': 0},
        CostLaw('V', 1.0, regime='slope tier, joint-Schur + REML-EM'),
        (32, 64, 128, 256, 512), challenging=True),
    ScalePath(
        'binomial-slope-agq',
        {'family': 'binomial', 'structure': 'unstructured', 'method': 'agq',
         'q': 16, 'n_per': 12, 'seed': 0, 'n_quad': 5},
        CostLaw('V', 1.0, regime='agq, n_quad^r tensor mode grid'),
        (32, 64, 128, 256), challenging=True),
]

# Brain-scale anchors (the measured headlines that validate the extrapolation;
# heavy GPU -- run in the deferred full-matrix step, `provisional` until then).
# Each carries its sweep path's label so the tool associates + validates it.
_ANCHORS = (
    {'family': 'gaussian', 'structure': 'intercept', 'method': 'pql',
     'q': 8, 'n_per': 24, 'seed': 0, 'V': 65536,
     'path': 'gaussian-intercept-few'},          # whole-brain voxel batch
    {'family': 'binomial', 'structure': 'intercept', 'method': 'pql',
     'V': 64, 'n_per': 8, 'seed': 0, 'q': 1024,
     'path': 'binomial-intercept-many'},         # 1024 per-subject intercepts
)

# drift / dev anchor: the cheapest, oracle-gated, non-challenging point.
_REPRESENTATIVE = {'family': 'gaussian', 'structure': 'intercept',
                   'method': 'pql', 'q': 8, 'n_per': 24, 'seed': 0, 'V': 64,
                   'path': 'gaussian-intercept-few'}

CASE = Case(
    name='glmm_fit',
    op_qualname='nitrix.stats.glmm.glmm_fit',
    tier='marquee',
    output_independent=True,           # each voxel an independent fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=scaling_sweep(_PATHS),
    representative=_REPRESENTATIVE,
    large_param_points=_ANCHORS,
    scale_paths=tuple(_PATHS),
    build=_build,
    complexity=(
        'mass-univariate GLMM by PQL over V voxels, dispatched on the level '
        'count q: few-tier (q<=64) dense per-voxel solve O((p+q)^3); '
        'many-tier (q>64) structured Schur O(N p^2 + q) -- LINEAR in q (the '
        'headline vs the dense looped R, ~O(q^3) per fit). Random slopes add '
        'the joint-Schur + REML-EM solver (the v3 robust path) or AGQ '
        '(n_quad^r tensor nodes -- a constant factor in V). Batched over V; R '
        'mgcv LOOPS one fit per voxel -> the batched-vs-looped speedup grows '
        'with V. HBM ~ V. Scale axes: V (batch, few-tier paths), q (level '
        'count, many-tier path).'),
    slow_baselines=(
        SlowBaseline(
            'R.mgcv',
            'per-voxel R mgcv::gam(s(g,bs="re"),method="REML") fits looped in '
            'one Rscript; seconds at V>=64, infeasible at brain scale. '
            'CPU-only; skip in dev cycles.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'R.iofloor',
            'no-op: returns zeros, so its rel_to_tol is large and MEANINGLESS '
            '-- the row exists only to time the CSV round-trip + R startup '
            'R.mgcv pays (economic subtracts the same-namespace floor).'),
    ),
)
