# -*- coding: utf-8 -*-
"""Tier-2 domain-tool ref: ``nitrix.smoothing.bilateral_gaussian`` vs ITK.

Edge-preserving bilateral smoothing. nitrix's op is a general *point-cloud*
bilateral (feature-metric, bounded neighbourhood); here it is configured as the
*image* special case to compare against **SimpleITK**'s
``BilateralImageFilter`` (the ITK engine ANTs is built on -- ANTsPy itself
exposes no bilateral). The match (see ``cases/_bilateral.py``): box window of
radius ``ceil(2.5*sigma_d)``, features ``[row, col, intensity]``,
``DiagonalMetric([sigma_d, sigma_d, sigma_r])``.

No fp64 oracle (``fp64_reference=None``): the **interior** matches sitk to
~1e-4 (window + both Gaussians + normalisation match exactly), but the r-pixel
**boundary** diverges (ITK's edge handling vs nitrix's replicate stencil), so
interior parity is asserted in ``tests/test_bilateral_cases.py`` instead.
GPU-native (semiring gather+reduce, no solver). Ratio vs SimpleITK. (No GPU
ref: cupy has no bilateral primitive.)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.smoothing import bilateral_gaussian

from ._base import BuiltPoint, Case
from ._bilateral import bilateral_image, grid_bilateral_setup, sitk_bilateral

_SIGMA_D, _SIGMA_R = 2.0, 0.2


def _build(param: Dict[str, Any]) -> BuiltPoint:
    h, w = param['shape']
    sd, sr = param.get('sigma_d', _SIGMA_D), param.get('sigma_r', _SIGMA_R)
    img = bilateral_image(h, w, param.get('seed', 0))
    vals, feats, ell, metric, _ = grid_bilateral_setup(img, sd, sr)
    jv = jax.block_until_ready(jnp.asarray(vals))
    jf = jax.block_until_ready(jnp.asarray(feats))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        # sitk filters the (h, w) image; nitrix takes (values, features).
        return (img,) if framework == 'numpy' else (jv, jf)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda v, f: bilateral_gaussian(
                v, f, metric=metric, neighbourhood=ell)),
        'simpleitk.Bilateral': ('simpleitk', sitk_bilateral(sd, sr)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='no fp64 oracle: interior matches sitk.Bilateral to '
                      '~1e-4 (window + domain/range Gaussians match); r-pixel '
                      'boundary diverges (ITK edge vs replicate). Asserted in '
                      'tests.',
        ratio_reference='nitrix-jax',
    )


# (height, width): cost ~ h*w * (2r+1)^2 (box-window gather + weighted reduce).
_SHAPES = [(64, 64), (128, 128), (256, 256)]

CASE = Case(
    name='bilateral_gaussian',
    op_qualname='nitrix.smoothing.bilateral_gaussian',
    output_independent=False,  # each output is a window-weighted average
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'sigma_d': _SIGMA_D, 'sigma_r': _SIGMA_R,
                   'seed': 0} for s in _SHAPES],
    representative={'shape': [128, 128], 'sigma_d': _SIGMA_D,
                    'sigma_r': _SIGMA_R, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
