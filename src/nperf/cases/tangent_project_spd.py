# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.linalg.tangent_project_spd`` vs nilearn/cupy.

The affine-invariant tangent-space projection of SPD matrices,
``log(R^-1/2 X R^-1/2)`` -- the connectome 'tangent space' embedding
(Varoquaux 2010) that flattens subject covariance matrices for linear
modelling. nitrix (jax) vs **nilearn**'s ``ConnectivityMeasure(kind=
'tangent')`` kernel (the canonical neuroimaging CPU floor -- the *first nilearn
reference in the suite*, and the only reference this op had) + a CuPy
eigh-based GPU reference, all scored against an fp64 eigh-based oracle on
well-conditioned SPD input.

Both matrix functions *consume* their eigh, so XLA lowers them off the broken
cuSOLVER path: **nitrix runs this on the GPU** even while the dense cuSOLVER
potrf/eigh path is wedged (see ``cases/_spd.py``). Ratio vs nilearn's tangent.

The nilearn floor runs with BLAS pinned to 1 thread (see ``_tangent.py``) --
its loop-of-small-eigh kernel churns a multi-threaded pool catastrophically
(a scheduler artifact, not its compute), and these eighs are too small to gain
from threads -- so the headline is the **GPU vs CPU-floor** comparison; the
within-CPU ratio reflects kernel+fusion, not a core-matched race.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.linalg import tangent_project_spd

from ._base import BuiltPoint, Case, to_cupy
from ._tangent import cupy_tangent, eig_tangent, nilearn_tangent, tangent_input


def _build(param: Dict[str, Any]) -> BuiltPoint:
    b, d = param['b'], param['d']
    Xs, R = tangent_input(b, d, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(Xs))
    jr = jax.block_until_ready(jnp.asarray(R))
    ref = eig_tangent(Xs, R)  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(Xs, R)
        return (Xs, R) if framework == 'numpy' else (jx, jr)

    baselines = {
        'nitrix-jax': ('jax', lambda x, r: tangent_project_spd(x, r)),
        'nilearn.tangent': ('nilearn', nilearn_tangent),  # CPU floor
        'cupy.eigh_tangent': ('cupy', cupy_tangent()),    # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (batch, d): cost ~ b · d³ (an eigh per matrix); output is (b, d, d).
_SHAPES = [(64, 64), (64, 128), (32, 256)]

CASE = Case(
    name='tangent_project_spd',
    op_qualname='nitrix.linalg.tangent_project_spd',
    # each output entry couples all eigenpairs of W·X·W -- not element-wise
    # independent, but the fp64 oracle is computed in full (§C, documentary).
    output_independent=False,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'b': b, 'd': d, 'seed': 0} for (b, d) in _SHAPES],
    representative={'b': 64, 'd': 128, 'seed': 0},
    build=_build,
    rtol=1e-3,
    # two composed matrix functions (sympower then symlog), so the fp32 error
    # compounds -- looser than symlog's 1e-3 atol (measured rel_to_tol ~0.5).
    atol=3e-3,
)
