# -*- coding: utf-8 -*-
"""Tier-2 augment: ``nitrix.augment.random_svf_displacement`` vs scipy / cupy.

Sample a smooth **diffeomorphic** displacement: a low-res random velocity
field, upsampled and integrated by scaling-and-squaring (the lab2im spatial-
corruption generator). The most compute-heavy augment (the integration is
``n_steps`` warp-compositions). A **generator** (no input image). **RNG op --
no cross-framework oracle** (`fp64_reference=None`): the ratio is a task-level
wall-clock comparison and the *property* (shape ``(*spatial, ndim)``, finite)
is checked in tests. cupy / numpy twins reuse the scaling-and-squaring via
`cupyx.scipy.ndimage` / `scipy.ndimage.map_coordinates` (cupy = GPU headline).
No MONAI analog (its elastic transforms are B-spline/grid, not velocity-field).
Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
from nitrix.augment import random_svf_displacement

from ._augment import cupy_svf, np_svf
from ._base import BuiltPoint, Case

_MAX_STD = 3.0
_GRID_FRACTION = 0.0625
_N_STEPS = 5


def _build(param: Dict[str, Any]) -> BuiltPoint:
    seed = int(param.get('seed', 0))
    shape = tuple(int(s) for s in param['shape'])
    max_std = float(param.get('max_std', _MAX_STD))
    gf = float(param.get('grid_fraction', _GRID_FRACTION))
    n_steps = int(param.get('n_steps', _N_STEPS))
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (key,) if framework == 'jax' else ()

    baselines = {
        'nitrix-jax': ('jax', lambda k: random_svf_displacement(
            shape, k, max_std=max_std, grid_fraction=gf, n_steps=n_steps)),
        'numpy.random_svf_displacement': (
            'numpy', np_svf(shape, max_std, gf, n_steps, seed)),
        'cupy.random_svf_displacement': (
            'cupy', cupy_svf(shape, max_std, gf, n_steps, seed)),  # GPU head
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG generator (scaling-and-squaring): no cross-frame '
                      'oracle; ratio is task-level wall-clock, the '
                      'shape/finite property is checked in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): low-res velocity -> upsample -> n_steps warp-compositions.
_SIZES = [48, 64, 96]

CASE = Case(
    name='random_svf_displacement',
    op_qualname='nitrix.augment.random_svf_displacement',
    output_independent=False,  # the integration couples the whole field
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'max_std': _MAX_STD,
                   'grid_fraction': _GRID_FRACTION, 'n_steps': _N_STEPS,
                   'seed': 0} for s in _SIZES],
    representative={'shape': [64, 64, 64], 'max_std': _MAX_STD,
                    'grid_fraction': _GRID_FRACTION, 'n_steps': _N_STEPS,
                    'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
