# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.stats.lme.flame_two_level`` vs FSL FLAME.

Voxelwise FLAME-style two-level group model: a single-parameter REML for the
between-subject variance ``sigma_b^2`` given **known** per-subject
within-variance, batched over all ``V`` voxels.  Output ``(V, 2)`` =
``[gamma, sigma_b^2]``.

**The fair competitor is the upstream tool, FSL FLAME** (``flameo
--runmode=flame1``): the model is *identical* (known-within-variance two-level
mixed effects), so flameo's ``pe1`` (gamma) and ``mean_random_effects_var1``
(sigma_b^2) match nitrix to ~1e-3 (validated).  flameo iterates
**voxel-by-voxel** on CPU while nitrix fits all ``V`` in ONE call -- the
batched-vs-looped story (cf. ``reml_fit`` vs statsmodels), the gap growing with
``V``.  A second looped-CPU baseline,
``statsmodels.stats.meta_analysis.combine_effects`` (the known-within-variance
two-level model **is** a random-effects meta-analysis: combine N per-subject
effects with known variances -> pooled effect + between-subject heterogeneity
``tau^2 = sigma_b^2``), is included; its ``tau^2`` is Paule-Mandel not REML, so
it diverges at the variance boundary -- a finding (surfaced via fidelity), not
a clean match.  Both external tools are CPU-only, looped -> ``slow_baselines``.

Fidelity is scored against a **closed-form** oracle (exact for the *constant*
within-variance case: the model covariance is ``(sigma_b^2 + s2) I`` and the
REML collapses to OLS + residual-variance; see ``cases/_lme.py``) -- matches
``flame_two_level`` to ~2e-4, the truth, **not** a perf baseline.  The
flameo wrapper is file-coupled (NIfTI + VEST design), so it pays a round-trip
nitrix doesn't -- subtracted via the ``fsl.iofloor`` no-op (economic_report).
GPU note: on THIS L4 nitrix's REML solver hits the cuSOLVER
``gpusolverDnCreate`` blocker at **all** V (recorded as a graceful skip,
``gpu_solver_unavailable`` -- measured across every stored run), so the op runs
**CPU-only** here; the headline is therefore nitrix's **batched-CPU** fit vs
the **looped-CPU** tools (flameo / statsmodels), not a GPU win.  ROOT CAUSE +
FIX filed: the 1x1 ``jnp.linalg.cholesky`` in the per-voxel ``vmap`` is the
blocker; a scalar closed-form path unblocks the GPU and is 3-6x faster on CPU
(nitrix FR ``lme-family-tiny-linalg-gpu-block-and-perf``; shared with
``reml_fit``).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.lme import flame_two_level

from ._base import ApproxBaseline, BuiltPoint, Case, SlowBaseline
from ._lme import (
    flame_closed_form,
    flame_input,
    flameo_flame1,
    flameo_iofloor,
    statsmodels_flame,
)

_S2 = 0.3  # constant known within-variance (enables the closed-form oracle)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, big_n = param['V'], param['N']
    beta, varw, x_group = flame_input(v, big_n, param.get('seed', 0), s2=_S2)
    jb = jax.block_until_ready(jnp.asarray(beta))
    jv = jax.block_until_ready(jnp.asarray(varw))
    jx = jax.block_until_ready(jnp.asarray(x_group))

    ref = flame_closed_form(beta.astype(np.float64),
                            x_group.astype(np.float64), _S2)  # (V, 2) oracle

    def _nitrix(b: Any, vw: Any, xg: Any) -> Any:
        r = flame_two_level(b, vw, xg)
        return jnp.stack([r.gamma_hat[:, 0], r.sigma_b_sq], -1)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        # all baselines take (beta, varw, x_group); jax gets device arrays
        return (jb, jv, jx) if framework == 'jax' else (beta, varw, x_group)

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        # FSL FLAME (flame1): the upstream tool, the fair competitor (looped).
        'fsl.flameo': ('fsl', flameo_flame1()),
        # the NIfTI round-trip flameo pays (subtracted by economic_report).
        'fsl.iofloor': ('fsl', flameo_iofloor()),
        # looped random-effects meta-analysis (Paule-Mandel tau^2; diverges
        # from REML at the boundary -- a finding, see docstring).
        'statsmodels.meta_analysis': ('statsmodels', statsmodels_flame()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (voxels, subjects): N subjects per voxel (typical fMRI group size); V scales
# the batch.  Larger V is where the device-resident batched fit pulls ahead.
_SHAPES = [(1024, 60), (8192, 60), (65536, 60)]
# Brain-volume scale.  (nitrix-jax skips on GPU at every tier here -- the
# cuSOLVER gpusolverDnCreate blocker, see complexity -- so these run on CPU.)
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
        'voxels: O(V * iters * N) -- linear in the voxel batch V. nitrix fits '
        'all V in ONE call; FSL FLAME (flameo) and statsmodels meta-analysis '
        'LOOP one fit per voxel, so the batched-vs-looped speedup GROWS with '
        'V (the headline, and why both are slow_baselines). MEASURED (L4): '
        'the GPU solver path hits the cuSOLVER gpusolverDnCreate blocker at '
        'ALL V (gpu_solver_unavailable -- a graceful skip, seen in every '
        'stored run), so nitrix runs CPU-only here; even so the batched CPU '
        'fit beats the looped CPU tools (~4x flameo at V=1024). HBM ~ V.'),
    build=_build,
    rtol=5e-3,  # iterative-solver convergence floor (lme design doc)
    atol=5e-3,
    slow_baselines=(
        SlowBaseline(
            'fsl.flameo',
            'FSL FLAME (flameo flame1) iterates voxel-by-voxel on CPU; the '
            'per-voxel mixed-effects fit x V voxels is minutes at brain '
            'scale. CPU-only; skip in dev cycles, run in the full matrix.'),
        SlowBaseline(
            'statsmodels.meta_analysis',
            'per-voxel combine_effects (Python loop) -- like statsmodels in '
            'reml_fit, ~ms/voxel, so V=65536+ is minutes. CPU-only; skip in '
            'dev cycles.'),
    ),
    # The iofloor is a NO-OP (timing only); statsmodels meta-analysis uses a
    # Paule-Mandel tau^2 (not REML) -> both have a fidelity that should be
    # REPORTED, not gated (else the row is refused and drops from the economic
    # join, losing the I/O floor / the batched-vs-looped timing).
    approximate_baselines=(
        ApproxBaseline(
            'fsl.iofloor',
            'no-op: returns a constant placeholder (gamma=sigma_b^2=0), so '
            'its rel_to_tol is large and MEANINGLESS -- the row exists only '
            'to time the NIfTI round-trip flameo pays (economic subtracts).'),
        ApproxBaseline(
            'statsmodels.meta_analysis',
            'gamma (the pooled GLS mean) matches the oracle; tau^2 is '
            'Paule-Mandel, NOT REML, so it diverges at the variance boundary '
            '(tau^2 -> 0) -- the divergence magnitude is the signal, read '
            'against the looped-vs-batched speed (measured ~0 at V<=8192).'),
    ),
)
