# -*- coding: utf-8 -*-
"""Tier-2 stats inference: ``nitrix.stats.inference.cluster_size_map`` vs
numpy / cupy.

Per-voxel cluster *size* (voxel count of the connected component it belongs
to) -- the cluster-extent statistic behind cluster-extent FWE. The labels are
generated **host-side** (``scipy.ndimage.label`` of ``stat > threshold``, 1-
connectivity) so nitrix and the references label the SAME clusters and only the
*size-map* computation is compared. Oracle: exact **numpy**
(``bincount`` over labels); GPU community bar: **cupy**
(``cupyx.scipy.ndimage`` + ``cupy.bincount``). Keyed ``shape`` (spatial volume
= scale axis). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.inference import cluster_size_map

from ._base import BuiltPoint, Case, to_cupy
from ._inference import (
    cupy_cluster_size,
    labels_from,
    np_cluster_size,
    stat_map,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(int(s) for s in param['shape'])
    stat = stat_map(shape, param.get('seed', 0))
    thr = float(np.percentile(stat, 95))  # several supra-thresh clusters
    labels = labels_from(stat, thr, conn=1)  # shared input (scipy-labelled)
    jlab = jax.block_until_ready(jnp.asarray(labels))
    ref = np_cluster_size()(labels)  # exact numpy size-map (fp64) = oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return (to_cupy(labels),)
        return (jlab,) if framework == 'jax' else (labels,)

    baselines = {
        'nitrix-jax': ('jax', lambda lab: cluster_size_map(lab)),
        'cupyx.ndimage.cluster_size': ('cupy', cupy_cluster_size()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
_LARGE = [(96, 96, 96), (128, 128, 128)]

CASE = Case(
    name='cluster_size_map',
    op_qualname='nitrix.stats.inference.cluster_size_map',
    output_independent=False,  # the component a voxel joins couples the volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': list(s), 'seed': 0} for s in _SHAPES],
    representative={'shape': [48, 48, 48], 'seed': 0},
    large_param_points=tuple({'shape': list(s), 'seed': 0} for s in _LARGE),
    complexity=(
        'a segmented count over the connected-component labels: O(V) given '
        'the labels (the labelling itself is shared host-side input). nitrix '
        'does it with one device-resident scatter-add; cupy with bincount. '
        'Scale axis = the spatial volume V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
