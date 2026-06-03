# -*- coding: utf-8 -*-
"""Tier-2 domain-tool ref: ``nitrix.geometry.resample`` vs ANTsPy / scipy.

Linear image resize (align_corners=True). The *genuine* ANTsPy in-memory op:
``ants.resample_image(..., interp_type=0)`` shares nitrix's exact convention
(verified to match to 0.0), so it is the canonical medical-imaging reference
here -- unlike ``apply_transforms`` (file-coupled) or bilateral (ANTsPy has
none). Plus ``scipy.ndimage.map_coordinates`` (CPU floor) + a CuPy GPU ref on
the same align_corners sample grid, scored against an fp64 oracle.

All samples are in-bounds, so there is **no boundary divergence** (cf. the warp
/ median / bilateral cases): a clean fp64 oracle that every baseline -- ANTs
included -- matches. Ratio vs the scipy resize. ANTs runs in its own refs env
(``NPERF_PYTHON_ANTS``); pure interpolation, so nitrix is GPU-pure.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.geometry import resample

from ._base import BuiltPoint, Case, to_cupy
from ._resample import ants_resample, cupy_resize, resize_coords, scipy_resize


def _build(param: Dict[str, Any]) -> BuiltPoint:
    in_shape = tuple(param['shape'])
    out_shape = tuple(param['out'])
    rng = np.random.default_rng(param.get('seed', 0))
    img = rng.standard_normal(in_shape + (1,)).astype(np.float32)
    img_j = jax.block_until_ready(jnp.asarray(img))
    coords = resize_coords(in_shape, out_shape)

    ref = scipy_resize(coords)(img.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(img)
        return (img_j,) if framework == 'jax' else (img,)

    baselines = {
        'nitrix-jax': ('jax', lambda im: resample(im, out_shape)),
        'ants.resample_image': (  # canonical domain-tool reference (ITK)
            'ants', ants_resample(out_shape)),
        'scipy.ndimage.map_coordinates': ('scipy', scipy_resize(coords)),
        'cupyx.scipy.ndimage.map_coordinates': (
            'cupy', cupy_resize(coords)),  # GPU on-target ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (in -> out): cost ~ output voxels * 2^ndim (linear gather); 2-D up to 3-D.
_SHAPES = [([128, 128], [256, 256]), ([256, 256], [512, 512]),
           ([64, 64, 64], [128, 128, 128])]

CASE = Case(
    name='resample',
    op_qualname='nitrix.geometry.resample',
    output_independent=True,  # each output samples a bounded input window
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'out': o, 'seed': 0} for (s, o) in _SHAPES],
    representative={'shape': [256, 256], 'out': [512, 512], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
