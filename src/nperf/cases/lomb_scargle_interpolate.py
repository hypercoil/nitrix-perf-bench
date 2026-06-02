# -*- coding: utf-8 -*-
"""Tier-2 (B11): ``nitrix.signal.lomb_scargle_interpolate`` vs joint-GLM.

Power-2014 motion-censoring interpolation: a **joint** masked least-squares fit
of the observed samples to a ``[DC | cos | sin]`` Lomb-Scargle basis, spliced
back through the observed samples (``where(mask, data, recon)``).  nitrix (jax,
batched over voxels with a shared mask) vs a from-scratch numpy joint-GLM
(CPU floor) + a CuPy joint-GLM (GPU ref).  See ``cases/_interp.py``.

**Fidelity is recorded inconclusive (`fp64_reference=None`).**  The masked Gram
is ill-conditioned (cond ~1e32); the rcond-truncated pseudo-inverse regularises
it, but the censored-frame reconstruction sits in the near-null-space where
fp32 / fp64 (and their trial grids) disagree on the truncation -- there is
no bit-level cross-impl truth there (SCHEMA §C).  The well-defined correctness
property -- the **observed-sample splice-through** (``recon[obs] == data[obs]``
exactly) -- is asserted in the tests instead.  (Use-case note for nitrix filed
in ``docs/feature-requests/perf-bench-feedback.md``: the filled values are a
spectral bridge for downstream AR/IIR filtering, not durable imputations.)

**GPU (eigh-family).** nitrix routes the K×K Gram eigh through ``safe_eigh`` ->
CPU on this L4, the batched solve stays on device (so nitrix runs on GPU,
hybrid).  The CuPy ref's ``cupy.linalg.eigh`` fails at K≥256 (recorded
``gpu_solver_unavailable``), so the apples-to-apples GPU bar holds at small
K (``obs`` ≲ 430).  Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.signal import lomb_scargle_interpolate

from ._base import BuiltPoint, Case, to_cupy
from ._interp import cupy_joint_glm, interp_input, joint_glm

_FID_NOTE = (
    'censored-frame recon is regularisation-sensitive (rcond-truncated '
    'pseudo-inverse of a ~1e32-conditioned masked Gram) -- no bit-level '
    'cross-impl truth; the observed-sample splice-through is exact and is '
    'verified in the tests instead (SCHEMA §C no-cross-impl-oracle).'
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    v, obs = param['V'], param['obs']
    data, mask = interp_input(v, obs, param.get('seed', 0))
    jd = jax.block_until_ready(jnp.asarray(data))
    jm = jax.block_until_ready(jnp.asarray(mask))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(data, mask)
        return (data, mask) if framework == 'numpy' else (jd, jm)

    baselines = {
        'nitrix-jax': ('jax', lambda d, m: lomb_scargle_interpolate(d, m)),
        'numpy.joint_glm': ('numpy', lambda d, m: joint_glm(d, m, np)),
        'cupy.joint_glm': ('cupy', cupy_joint_glm()),  # GPU ref (fails K≥256)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None, ratio_reference='nitrix-jax',
        fidelity_note=_FID_NOTE,
    )


# (voxels, observations): shared censor mask; obs sets K = 2·n_freq+1 (the eigh
# dim (capped by censoring_budget); the CuPy ref fails once K≥256 (obs≳430).
_SHAPES = [(4096, 256), (4096, 512), (4096, 1024)]

CASE = Case(
    name='lomb_scargle_interpolate',
    op_qualname='nitrix.signal.lomb_scargle_interpolate',
    output_independent=True,  # each voxel's fit is independent (shared basis)
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'V': v, 'obs': o, 'seed': 0} for (v, o) in _SHAPES],
    representative={'V': 4096, 'obs': 512, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
