# -*- coding: utf-8 -*-
"""PERF_AUDIT port (B11): ``nitrix.geometry.spatial_transform`` vs scipy.

Linear-interpolation resampling by an absolute-coordinate deformation field --
nitrix vs ``scipy.ndimage.map_coordinates`` (same convention: absolute sample
coords).  They agree to interpolation round-off **for in-bounds samples**; only
out-of-bounds fractional samples diverge (boundary interpolation differs).  So
the deformation is kept strictly in-bounds (a documented, fair choice -- the
interpolation work, hence the perf, is unchanged), making scipy a clean fp64
oracle.  Ratio via ``--reference scipy.ndimage.map_coordinates``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage as spnd
from nitrix.geometry import spatial_transform

from ._base import BuiltPoint, Case


def _scipy_map(img: np.ndarray, deform: np.ndarray) -> np.ndarray:
    '''map_coordinates on the single channel, returned channel-last (H,W,1).'''
    coords = deform.transpose(2, 0, 1)  # (ndim, H, W)
    out = spnd.map_coordinates(img[..., 0], coords, order=1, mode='constant')
    return out[..., None]


def _build(param: Dict[str, Any]) -> BuiltPoint:
    h, w = param['shape']
    rng = np.random.default_rng(param.get('seed', 0))
    img = rng.standard_normal((h, w, 1)).astype(np.float32)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # In-bounds absolute-coord deformation (clip to [1, size-2]) so nitrix and
    # scipy don't diverge on out-of-bounds boundary handling.
    di = np.clip(ii + 0.3 * rng.standard_normal((h, w)), 1, h - 2)
    dj = np.clip(jj + 0.3 * rng.standard_normal((h, w)), 1, w - 2)
    deform = np.stack([di, dj], axis=-1).astype(np.float32)
    img_j = jax.block_until_ready(jnp.asarray(img))
    def_j = jax.block_until_ready(jnp.asarray(deform))

    # fp64 oracle: scipy map_coordinates in double, channel-last.
    ref = _scipy_map(img.astype(np.float64), deform.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (img, deform) if framework == 'numpy' else (img_j, def_j)

    baselines = {
        'nitrix-jax': (
            'jax',
            lambda im, df: spatial_transform(im, df, mode='constant'),
        ),
        'scipy.ndimage.map_coordinates': ('scipy', _scipy_map),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [[64, 64], [256, 256]]

CASE = Case(
    name='spatial_transform',
    op_qualname='nitrix.geometry.spatial_transform',
    output_independent=True,  # each output samples a bounded neighbourhood
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [256, 256], 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
