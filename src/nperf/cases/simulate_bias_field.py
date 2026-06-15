# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.simulate_bias_field`` vs scipy / cupy.

Generate a smooth multiplicative intensity non-uniformity field: a low-res
Gaussian control grid (``grid_fraction``·shape), smoothly upsampled to the full
volume and exponentiated (-> positive). A **generator** (no input image).
**RNG op -- no cross-framework oracle** (`fp64_reference=None`): the ratio is a
task-level wall-clock comparison and the *property* (positive, finite, full
shape) is checked in tests. cupy (`cupyx.scipy.ndimage.zoom`) is the GPU
headline ref; numpy (`scipy.ndimage.zoom`) the CPU floor. No MONAI row --
`RandBiasField` *applies* a polynomial bias to an input image (different I/O),
so it is not a twin for field *generation*. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
from nitrix.augment import simulate_bias_field

from ._augment import cupy_bias_field, np_bias_field
from ._base import BuiltPoint, Case

_MAX_STD = 0.5
_GRID_FRACTION = 0.04


def _build(param: Dict[str, Any]) -> BuiltPoint:
    seed = int(param.get('seed', 0))
    shape = tuple(int(s) for s in param['shape'])
    max_std = float(param.get('max_std', _MAX_STD))
    gf = float(param.get('grid_fraction', _GRID_FRACTION))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        # a generator: jax needs the key; the numpy/cupy twins close over the
        # shape + seed and take no input.
        return (key,) if framework == 'jax' else ()

    baselines = {
        'nitrix-jax': ('jax', lambda k: simulate_bias_field(
            shape, k, max_std=max_std, grid_fraction=gf)),
        'numpy.simulate_bias_field': (
            'numpy', np_bias_field(shape, max_std, gf, seed)),  # scipy zoom
        'cupy.simulate_bias_field': (
            'cupy', cupy_bias_field(shape, max_std, gf, seed)),  # GPU headline
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG generator: no cross-framework oracle; ratio is '
                      'task-level wall-clock, the positive/finite/shape '
                      'property is checked in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): low-res grid -> cubic upsample to n^3.
_SIZES = [64, 96, 128]

CASE = Case(
    name='simulate_bias_field',
    op_qualname='nitrix.augment.simulate_bias_field',
    output_independent=False,  # the upsampled field couples neighbours
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'max_std': _MAX_STD,
                   'grid_fraction': _GRID_FRACTION, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'max_std': _MAX_STD,
                    'grid_fraction': _GRID_FRACTION, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
