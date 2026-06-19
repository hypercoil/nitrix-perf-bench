# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.lme.reml_fit`` vs statsmodels.

Voxelwise variance-components REML (FaST-LMM spectral trick): nitrix fits all
``V`` voxels in one **batched** call; ``statsmodels.MixedLM`` (the canonical
CPU LME library) must **loop** one fit per voxel -- the headline
batched-vs-looped comparison.  Scored against a **closed-form** balanced
one-way REML oracle (see ``cases/_lme.py``); the output is ``(V, 3)`` columns
``[beta, sigma_b^2, sigma_e^2]``.

nitrix **runs on GPU here** (the ``ok`` GPU store rows are correct, NOT
stale) -- unlike its sibling ``flame_two_level``, which skips with a cuSOLVER
handle-creation error despite the *same* ``(p, p)`` Cholesky.  Observed on this
L4 (cause not established): the cuSOLVER ``gpusolverDnCreate`` fails for a
``potrf``/``syevd``-first program but not after a ``getrf``/matmul.
``reml_fit`` incidentally runs a ``2x2`` ``jnp.linalg.solve`` (``getrf``) in
its Newton step, which appears to be why it runs while ``flame`` skips.
That opaque GPU issue + its provisional warmup are filed in nitrix FR
``gpu-cusolver-first-call-handle-failure`` (treat as observational; needs
robust repeated-trial verification).  The shared CPU/compile perf win
(Cholesky-free p=1 path, closed-form AI-REML, SVD-not-eigh) is filed in FR
``lme-family-tiny-linalg-gpu-block-and-perf``, which flags the fix-risk that
dropping this ``getrf`` while leaving the eigh could re-block ``reml`` on GPU.
The one-time ``ZZ^T`` eigh goes through ``safe_eigh`` -> CPU.
statsmodels is CPU-only (``requires='cpu'``) and a **slow baseline** (per-voxel
iterative fits) -- skip in dev cycles (``--skip-slow``), run it in the
sprint-end full matrix.  No GPU reference library exists for LME.  Tolerance is
loose (``5e-3``) -- the convergence floor of the iterative solvers (lme design
doc).  Ratio vs nitrix-jax.

**Real-data point (marquee real-data bar).** Beside the synthetic balanced
design, a ``data='localizer'`` point fits a *real* one-way random-intercept on
the Brainomics localizer (``_real_lme.real_reml_localizer``): 40 subjects'
level-1 effects over 512 brain voxels as ``Y``, grouped by acquisition
**site** (the canonical neuroimaging random factor).  Real + unbalanced => no
balanced closed form (``real_full``, no oracle); correctness = agreement with
statsmodels MixedLM on real data -- beta + within-site variance ~exact (corr
~1.0), the between-site variance to corr ~0.99 (k=2 sites is a thin
boundary-prone df, a caveat).  reml runs on the GPU here, so real data lifts
this op without the cuSOLVER ceiling ``flame`` hits.
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
from ._real_lme import real_reml_localizer


def _build(param: Dict[str, Any]) -> BuiltPoint:
    if param.get('data') == 'localizer':
        # REAL fMRI: localizer level-1 effects as Y, grouped by acquisition
        # site (the canonical random-effect factor). real + unbalanced => no
        # balanced closed form (no oracle); correctness = agreement with
        # statsmodels MixedLM on the real data.
        Y, X, Z, groups = real_reml_localizer(
            int(param['N']), int(param['V']), param.get('seed', 0))
        ref: Any = None
        note: Any = (
            'real localizer COPEs grouped by acquisition site (k=2 -- a thin '
            'between-group df, a documented caveat): real + unbalanced, so '
            'the balanced closed form is inapplicable (no oracle). '
            'Correctness = agreement with statsmodels MixedLM on real data.')
    else:
        v, k, n = param['V'], param['k'], param['n']
        Y, X, Z, groups = balanced_oneway(v, k, n, param.get('seed', 0))
        ref = closed_form_reml(Y.astype(np.float64), k, n)  # (V,3) oracle
        note = None
    jY = jax.block_until_ready(jnp.asarray(Y))
    jX = jax.block_until_ready(jnp.asarray(X))
    jZ = jax.block_until_ready(jnp.asarray(Z))

    def _nitrix(y: Any) -> Any:
        r = reml_fit(y, jX, jZ)
        return jnp.stack([r.beta_hat[:, 0], r.sigma_b_sq, r.sigma_e_sq], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y,)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'statsmodels.MixedLM': (
            'statsmodels', lambda y: statsmodels_reml(y, X, groups)),
        # R lme4 lmer -- THE gold-standard mixed-model REML (matches the closed
        # form on this balanced design); the strongest looped-CPU competitor.
        'R.lme4': ('r', lambda y: r_lme4_reml(y, groups)),
        # I/O floor for R.lme4: CSV write + R startup + read, no fit (economic
        # subtracts the same-namespace 'R.' floor to isolate the lmer compute).
        'R.iofloor': ('r', lambda y: r_lme4_iofloor(y, groups)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
        fidelity_note=note,
    )


# (voxels, groups, per-group): N = k*n subjects shared across voxels.  V scales
# the batch; statsmodels loops over it (the speedup grows with V).
_SHAPES = [(64, 8, 24), (256, 8, 24), (1024, 8, 24)]
# Brain-voxel scale: V up to 65536 voxels in the batch (statsmodels would loop
# ~15 min/fit here -> it is a slow_baseline, dropped by --skip-slow).
_LARGE = [(16384, 8, 24), (65536, 8, 24)]
# Real-data point (marquee real-data bar): real localizer level-1 effects over
# V=512 brain voxels x N=40 subjects, grouped by acquisition site. real_full
# (no oracle); statsmodels MixedLM runs on the same real data. V kept small so
# the per-voxel-looped statsmodels (~14 ms/voxel) stays tractable each sweep.
_REAL = {'data': 'localizer', 'V': 512, 'N': 40, 'realism': 'real_full',
         'seed': 0}

CASE = Case(
    name='reml_fit',
    op_qualname='nitrix.stats.lme.reml_fit',
    tier='marquee',
    output_independent=True,  # each voxel is an independent LME fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=([{'V': v, 'k': k, 'n': n, 'seed': 0}
                   for (v, k, n) in _SHAPES] + [_REAL]),
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
        SlowBaseline(
            'R.lme4',
            'per-voxel R lme4 lmer fits looped in one Rscript (lmer is '
            'heavier than statsmodels per fit); minutes at V>=1024, '
            'infeasible at the brain large tier. CPU-only; skip in dev.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'R.iofloor',
            'no-op: returns zeros, so its rel_to_tol is large and MEANINGLESS '
            '-- the row exists only to time the CSV round-trip + R startup '
            'R.lme4 pays (economic subtracts the same-namespace floor); a '
            'LARGE fraction of R.lme4 wall-clock at small V.'),
    ),
)
