# -*- coding: utf-8 -*-
"""Tier-2 stats inference: ``nitrix.stats.inference.tfce`` vs numpy / cupy.

Threshold-free cluster enhancement (Smith-Nichols 2009): integrate
``extent(h)^E * h^H`` over a ladder of thresholds ``h``, turning a statistic
image into a cluster-enhanced one *without* a hard cluster-forming threshold.
The per-threshold connected-component labelling is the cost; nitrix runs the
whole ladder on-device behind one compile, while the reference re-labels on the
host per step.

Benched **one-sided** (``two_sided=False``) so it matches the exact **numpy**
fp64 oracle (``_inference.np_tfce`` -- a faithful Smith-Nichols reimpl that
integrates ``extent^E * h^H * dh`` over ``n_steps`` thresholds; matches nitrix
to ~1e-7). GPU community bar: a **cupy** TFCE over
``cupyx.scipy.ndimage.label`` (same ladder, on the GPU). There is no standalone
*CPU* community TFCE tool (TFCE ships inside FSL ``randomise`` -- benched in
``permutation_test``), so the economic read falls back to GPU-vs-own-CPU.
Keyed ``shape`` (the spatial volume = the scale axis). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.stats.inference import tfce

from ._base import BuiltPoint, Case, to_cupy
from ._inference import cupy_tfce, np_tfce, stat_map


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(int(s) for s in param['shape'])
    stat = stat_map(shape, param.get('seed', 0))
    jstat = jax.block_until_ready(jnp.asarray(stat))
    ref = np_tfce()(stat)  # numpy Smith-Nichols TFCE (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return (to_cupy(stat),)
        return (jstat,) if framework == 'jax' else (stat,)

    baselines = {
        # one-sided to match the one-sided oracle (positive blobs in stat_map).
        'nitrix-jax': ('jax', lambda s: tfce(s, two_sided=False)),
        'cupyx.ndimage.tfce': ('cupy', cupy_tfce()),  # GPU community bar
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
_LARGE = [(96, 96, 96)]

CASE = Case(
    name='tfce',
    op_qualname='nitrix.stats.inference.tfce',
    output_independent=False,  # the threshold ladder couples the whole volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': list(s), 'seed': 0} for s in _SHAPES],
    representative={'shape': [48, 48, 48], 'seed': 0},
    large_param_points=tuple({'shape': list(s), 'seed': 0} for s in _LARGE),
    complexity=(
        'integrate extent(h)^E * h^H over n_steps (=100) thresholds: each '
        'step is a connected-component labelling over prod(shape) voxels, so '
        'O(n_steps * V). nitrix runs the whole ladder on-device behind one '
        'compile; the numpy/cupy references re-label per step. HBM ~ V (a few '
        'working buffers). Scale axis = the spatial volume V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
