# -*- coding: utf-8 -*-
"""Tier-2 morphology: ``nitrix.morphology.distance_transform`` (chamfer).

The opt-in **chamfer** branch (B18 Win 1), measured independently of the
euclidean default (``distance_transform`` case).  ``metric='chebyshev'`` runs
the iterated ``TROPICAL_MIN_PLUS`` convolution (3x3 chessboard step kernel,
``max(spatial)`` iterations) -- **exact for the chessboard metric**, so the
reference is ``scipy.ndimage.distance_transform_cdt(metric='chessboard')``
(also exact) and the gate is tight (matches *exactly*; chamfer distances are
integers).  This is the iterative path B18 flagged as ~80x slower than the
euclidean engine on GPU, so it is measured *as* the iterative engine it is, not
folded into the euclidean win.

No on-target GPU reference: ``cupyx.scipy.ndimage`` implements
``distance_transform_edt`` but **not** ``distance_transform_cdt``, so the GPU
column here is nitrix-only (the CPU floor + oracle is scipy ``cdt``).  Ratio vs
``nitrix-jax``.  (``metric='city_block'`` is the analogous taxicab chamfer vs
``cdt(metric='taxicab')``; chebyshev is the representative chamfer here.)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import distance_transform

from ._base import BuiltPoint, Case
from ._distance import blob_mask, scipy_cdt

_METRIC = 'chebyshev'  # chessboard chamfer (exact for L-inf)
_CDT = 'chessboard'    # matching scipy distance_transform_cdt metric


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    mask = blob_mask(shape, param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(mask))

    ref = scipy_cdt(_CDT)(mask)  # exact chessboard chamfer (oracle)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (mask,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: distance_transform(m, metric=_METRIC)),
        'scipy.ndimage.distance_transform_cdt': (
            'scipy', scipy_cdt(_CDT)),  # CPU floor + oracle
        # no cupy GPU ref: cupyx.scipy.ndimage has no distance_transform_cdt.
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# Modest sizes: the chamfer engine iterates max(spatial) times (a sequential
# fori_loop of 3x3 min-plus convs), so the iteration count -- not just the grid
# -- drives cost; large grids belong to the euclidean case.
_SHAPES = [[64, 64], [128, 128], [64, 64, 64]]

CASE = Case(
    name='distance_transform_chamfer',
    op_qualname='nitrix.morphology.distance_transform',
    output_independent=False,  # each output is a global min over the mask
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': s, 'seed': 0} for s in _SHAPES],
    representative={'shape': [128, 128], 'seed': 0},
    build=_build,
    # chebyshev chamfer is EXACT for the chessboard metric -> matches cdt
    # exactly (integer distances); tight gate.
    rtol=1e-3,
    atol=1e-4,
)
