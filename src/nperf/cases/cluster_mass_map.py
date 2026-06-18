# -*- coding: utf-8 -*-
"""Tier-2 stats inference: ``nitrix.stats.inference.cluster_mass_map`` vs
numpy / cupy.

Per-voxel cluster *mass*: the component's summed excess statistic
``sum(max(stat - threshold, 0))``, written to each member voxel -- the cluster-
mass statistic behind cluster-mass FWE (and the alternative to TFCE in FSL
``randomise``). Labels are generated **host-side** (``scipy.ndimage.label`` of
``stat > threshold``, 1-connectivity) so nitrix and the references operate on
the SAME clusters and only the mass computation is compared. Oracle: exact
**numpy** (segmented weighted sum); GPU community bar: **cupy** (``bincount``
with weights). There is no standalone CPU community mass tool (cluster mass is
computed inside FSL ``randomise`` -- benched in ``permutation_test``), so the
economic read falls back to GPU-vs-own-CPU. Keyed ``shape`` (spatial volume =
scale axis). Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.inference import cluster_mass_map

from ._base import BuiltPoint, Case, to_cupy
from ._inference import (
    cupy_cluster_mass,
    labels_from,
    np_cluster_mass,
    stat_map,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(int(s) for s in param['shape'])
    stat = stat_map(shape, param.get('seed', 0))
    thr = float(np.percentile(stat, 95))
    labels = labels_from(stat, thr, conn=1)  # shared input (scipy-labelled)
    jlab = jax.block_until_ready(jnp.asarray(labels))
    jstat = jax.block_until_ready(jnp.asarray(stat))
    ref = np_cluster_mass(thr)(labels, stat)  # exact numpy mass (fp64) oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return (to_cupy(labels), to_cupy(stat))
        return (jlab, jstat) if framework == 'jax' else (labels, stat)

    baselines = {
        'nitrix-jax': ('jax', lambda lab, s: cluster_mass_map(lab, s, thr)),
        'cupyx.ndimage.cluster_mass': ('cupy', cupy_cluster_mass(thr)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
_LARGE = [(96, 96, 96), (128, 128, 128)]

CASE = Case(
    name='cluster_mass_map',
    op_qualname='nitrix.stats.inference.cluster_mass_map',
    output_independent=False,  # the component a voxel joins couples the volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': list(s), 'seed': 0} for s in _SHAPES],
    representative={'shape': [48, 48, 48], 'seed': 0},
    large_param_points=tuple({'shape': list(s), 'seed': 0} for s in _LARGE),
    complexity=(
        'a segmented weighted sum (excess = max(stat-thr,0)) over the CC '
        'labels: O(V) given the labels (shared host-side input). '
        'nitrix does it with one device-resident scatter-add; cupy with a '
        'weighted bincount. HBM ~ V. Scale axis = the spatial volume V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
