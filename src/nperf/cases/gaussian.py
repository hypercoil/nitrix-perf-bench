# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.smoothing.gaussian`` vs scipy.ndimage.

Gaussian smoothing -- nitrix on jax vs ``scipy.ndimage.gaussian_filter`` (host;
the ``scipy`` provider).  scipy is in the base env, so no refs env is needed.
The two agree to fp32 round-off, so ``scipy.ndimage.gaussian_filter`` in fp64
on the same values is the oracle and both baselines pass the gate.  Ratio via
``--reference scipy.ndimage.gaussian_filter``.

**No SimpleITK floor here (deliberate).**  Unlike erode/dilate/median/
distance_transform (where ITK matches the scipy oracle and is added as the
imaging-standard floor), ITK's Gaussian filters use a *different kernel* from
scipy's sampled, truncated continuous Gaussian: ``DiscreteGaussian`` is
Lindeberg's discrete-Gaussian (modified-Bessel) kernel and
``SmoothingRecursiveGaussian`` is a Deriche IIR approximation -- both diverge
from the oracle far past tolerance (measured rel_to_tol ~2e3, max ~0.36 at
sigma=1.5), so neither is a fair fp64-oracle floor.  scipy / cupyx remain the
references.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.smoothing import gaussian

from ._base import BuiltPoint, Case, to_cupy


def _cupy_gaussian(x: Any, sigma: float) -> Any:
    '''GPU gaussian_filter (cupyx.scipy.ndimage); cupy lazy (refs-cupy env).'''
    from cupyx.scipy import ndimage as cnd

    return cnd.gaussian_filter(x, sigma=sigma)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    sigma = param.get('sigma', 1.5)
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal(shape).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))

    # fp64 oracle.
    ref = spnd.gaussian_filter(X.astype(np.float64), sigma=sigma)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: gaussian(x, sigma=sigma)),
        'scipy.ndimage.gaussian_filter': (
            'scipy', lambda x: spnd.gaussian_filter(x, sigma=sigma)),
        'cupyx.scipy.ndimage.gaussian_filter': (
            'cupy', lambda x: _cupy_gaussian(x, sigma)),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# 2-D and 3-D fields (the PERF_AUDIT ladder); the 3-D point is where the
# separable-vs-dense gap is widest.
_SHAPES = [[64, 64], [256, 256], [64, 64, 64]]

CASE = Case(
    name='gaussian',
    op_qualname='nitrix.smoothing.gaussian',
    output_independent=False,  # each output mixes a sigma-neighbourhood
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'sigma': 1.5, 'seed': 0} for s in _SHAPES],
    representative={'shape': [64, 64, 64], 'sigma': 1.5, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
