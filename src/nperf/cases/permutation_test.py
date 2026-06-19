# -*- coding: utf-8 -*-
"""Tier-2 HEADLINE: ``inference.permutation_test`` vs FSL randomise + nilearn.

THE marquee inference op: a TFCE permutation test with family-wise-error
control (the neuroimaging group-stats workhorse). A one-sample (sign-flip)
design over ``N`` subjects on a ``*spatial`` volume; ``n_perm`` permutations
each refit the GLM t-contrast, TFCE-enhance, and take the spatial max to build
the FWE null. **nitrix batches the whole permutation loop on the GPU behind one
compile**, while the community gold standards LOOP the permutations on CPU --
so the economic gap grows with ``n_perm * voxels`` (a test that is minutes on
CPU becomes seconds on the GPU; the central multiplicative story).

Baselines (both TFCE permutation, both CPU loops -> ``slow_baselines``):
**FSL** ``randomise -1 -T`` (THE gold standard; reads ``*_tfce_corrp_tstat1`` =
``1 - p``) + its NIfTI-round-trip ``fsl.iofloor``; and **nilearn**
``permuted_ols(tfce=True)`` (the neuro-Python tool, via a full-volume masker).

**Fidelity is Monte Carlo, so there is NO exact oracle**
(``fp64_reference=None``): the FWE p-map is a permutation estimate, and nitrix
(a fixed jax key) vs randomise / nilearn (their own RNG / permutation sets)
agree only *statistically* (validated out-of-band by spatial correlation of the
p-maps + recovery of the planted cluster -- like ``flame_two_level``'s
real-data point). The DETERMINISTIC observed ``stat`` t-map IS gated against a
numpy one-sample-t oracle in the tests. Keyed ``{shape, subj, n_perm}`` (scale
axis = ``n_perm * voxels``; uses
``subj`` not ``N`` to avoid the bbr key clash). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.inference import permutation_test

from ._base import ApproxBaseline, BuiltPoint, Case, SlowBaseline
from ._inference import (
    fsl_randomise,
    fsl_randomise_iofloor,
    nilearn_permuted_ols,
    perm_data,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(int(s) for s in param['shape'])
    subj = int(param['subj'])
    n_perm = int(param['n_perm'])
    seed = int(param.get('seed', 0))
    data, design, contrast = perm_data(shape, subj, seed)
    jd = jax.block_until_ready(jnp.asarray(data))
    jdes = jax.block_until_ready(jnp.asarray(design))
    jcon = jax.block_until_ready(jnp.asarray(contrast))
    key = jax.random.PRNGKey(seed)

    def _nitrix(d: Any, des: Any, con: Any) -> Any:
        return permutation_test(d, des, con, key=key, n_perm=n_perm,
                                enhancement='tfce').p_fwe

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return ((jd, jdes, jcon) if framework == 'jax'
                else (data, design, contrast))

    baselines = {
        'nitrix-jax': ('jax', _nitrix),
        'fsl.randomise': ('fsl', fsl_randomise(n_perm)),       # gold standard
        'fsl.iofloor': ('fsl', fsl_randomise_iofloor()),     # round-trip floor
        'nilearn.permuted_ols': ('nilearn', nilearn_permuted_ols(n_perm)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None, ratio_reference='nitrix-jax',
        fidelity_note=(
            'permutation FWE p-maps are Monte Carlo: nitrix uses a fixed jax '
            'key, FSL randomise / nilearn their own RNG + permutation sets, '
            'so the maps agree statistically (spatial correlation + planted-'
            'cluster recovery, validated out-of-band), NOT bit-exactly. The '
            'deterministic observed t-map (PermResult.stat) matches a numpy '
            'one-sample-t oracle (gated in the tests).'),
    )


# (nx, ny, nz, subj, n_perm): dev tier kept small (the representative anchors
# drift -- nitrix runs n_perm permutations under one jit there).
_DEV = [(24, 24, 24, 16, 100), (32, 32, 32, 20, 200)]
# Scale tier (the headline): grow n_perm (the permutation loop) at fixed
# volume, then a volume step -- where batched-GPU-vs-looped-CPU compounds. FSL
# randomise / nilearn loop these on CPU (minutes) -> slow_baselines.
_LARGE = [(32, 32, 32, 20, 500), (32, 32, 32, 20, 1000),
          (48, 48, 48, 20, 500)]


def _pt(nx: int, ny: int, nz: int, subj: int, n_perm: int) -> Dict[str, Any]:
    return {'shape': [nx, ny, nz], 'subj': subj, 'n_perm': n_perm, 'seed': 0}


CASE = Case(
    name='permutation_test',
    op_qualname='nitrix.stats.inference.permutation_test',
    tier='marquee',
    output_independent=False,  # the spatial-max FWE null couples the volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[_pt(*p) for p in _DEV],
    representative=_pt(24, 24, 24, 16, 100),
    large_param_points=tuple(_pt(*p) for p in _LARGE),
    complexity=(
        'n_perm permutations, each a GLM t-contrast + TFCE enhancement + '
        'spatial max over prod(shape) voxels: O(n_perm * V * tfce_steps). '
        'nitrix batches the whole loop on-device (vmap/scan) behind ONE '
        'compile; FSL randomise and nilearn permuted_ols LOOP the perms on '
        'CPU, so the batched-vs-looped speedup GROWS with n_perm * V (the '
        'headline). HBM ~ V * (a few permutation buffers). Scale axis = '
        'n_perm * V.'),
    build=_build,
    rtol=1e-3,  # only the deterministic stat is gated (in tests); p_fwe is MC
    atol=1e-4,
    slow_baselines=(
        SlowBaseline(
            'fsl.randomise',
            'FSL randomise loops n_perm permutations on CPU (each a GLM + '
            'TFCE over the volume); minutes at n_perm>=500 / brain scale. '
            'CPU-only; skip in dev cycles, run in the full matrix.'),
        SlowBaseline(
            'nilearn.permuted_ols',
            'nilearn permuted_ols loops the permutations on CPU (joblib); '
            'TFCE + masker makes it minutes at scale. CPU-only; skip in dev.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'fsl.iofloor',
            'no-op: returns zeros, so its rel_to_tol is large and MEANINGLESS '
            '-- the row exists only to time the 4D NIfTI round-trip randomise '
            'pays (economic subtracts the same-namespace floor).'),
    ),
)
