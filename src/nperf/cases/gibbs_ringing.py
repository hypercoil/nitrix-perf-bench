# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.gibbs_ringing`` vs numpy / cupy.

Gibbs (truncation) ringing -- the oscillation near sharp edges from a sharply
band-limited acquisition. nitrix transforms to k-space, **hard-zeroes** every
frequency whose normalised radius exceeds ``(1 - alpha)`` of the max, and
inverse-transforms. FFT-bound (a real GPU-FFT story, not just elementwise).

References: nitrix (jax) vs the numpy reimplementation (the exact same k-space
truncation, fp64 oracle) and a **cupy GPU reference** (the on-target FFT
headline bar). Ratio vs nitrix-jax.

**No MONAI baseline -- an honest divergence (a finding, not a forced ref).**
MONAI's `GibbsNoise` models a *different* artifact: a **soft** radial roll-off
of k-space, where nitrix applies a **hard** spherical cutoff (the sharp
truncation is what actually produces ringing). On a real-range volume the two
diverge ~0.18 -- a different transform, not a small tradeoff -- so MONAI is not
apples-to-apples here (cf. the percentile-clip / ANTs `TruncateImageIntensity`
gap). The bar is the numpy fp64 oracle + the cupy GPU ref.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import gibbs_ringing

from ._augment import augment_input, cupy_gibbs, np_gibbs
from ._base import BuiltPoint, Case, to_cupy

_ALPHA = 0.3  # truncate the outer 30% of the k-space radius


def _build(param: Dict[str, Any]) -> BuiltPoint:
    alpha = float(param.get('alpha', _ALPHA))
    X = augment_input(param['shape'], param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_gibbs(alpha)(X.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: gibbs_ringing(x, alpha)),
        'numpy.gibbs_ringing': ('numpy', np_gibbs(alpha)),  # CPU floor/oracle
        'cupy.gibbs_ringing': ('cupy', cupy_gibbs(alpha)),  # GPU FFT headline
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (cube side): an n^3 FFT pair + a radial mask multiply; cost ~ n^3 log n.
_SIZES = [64, 96, 128]

CASE = Case(
    name='gibbs_ringing',
    op_qualname='nitrix.augment.gibbs_ringing',
    output_independent=False,  # the FFT couples the whole volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'alpha': _ALPHA, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'alpha': _ALPHA, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
