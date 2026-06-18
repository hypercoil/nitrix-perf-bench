# -*- coding: utf-8 -*-
"""Tier-2 inference CHAIN: the cluster-extent pipeline vs FSL ``fsl-cluster``.

The atomic ``cluster_size_map`` case benches the size-map step in isolation,
but **no community tool exposes that step alone** -- FSL/AFNI FUSE the cluster-
forming threshold, the connected-component labelling, and the per-voxel size
into ONE pass.  So the fair, realistic comparison (the chain-parity rule) is
the nitrix CHAIN

    cluster_size_map(supra_threshold_clusters(stat, thr))

run under a **single JIT** (the harness wraps the baseline in ``jax.jit``, so
the threshold->label->size pipeline fuses behind one compile) against the
community **bundle** FSL ``fsl-cluster --in --thresh --osize`` (threshold +
CC-label + size image, one call).  ``--connectivity=6`` matches nitrix
``connectivity=1`` (scipy 6-neighbour); the size map is **label-permutation-
invariant**, so it is directly comparable to the exact **numpy** oracle (scipy
``label`` + ``bincount``, fp64).  fsl-cluster is CPU-only + file-coupled (NIfTI
round-trip) -> a
``slow_baseline``; the ``fsl.iofloor`` no-op times the round-trip nitrix never
pays (economic_report subtracts it).  Keyed ``shape`` (spatial volume = scale
axis).  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.stats.inference.cluster import (
    cluster_size_map,
    supra_threshold_clusters,
)

from ._base import ApproxBaseline, BuiltPoint, Case, SlowBaseline
from ._inference import (
    fsl_cluster_iofloor,
    fsl_cluster_osize,
    np_cluster_extent,
    stat_map,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(int(s) for s in param['shape'])
    stat = stat_map(shape, param.get('seed', 0))
    thr = float(np.percentile(stat, 95))  # cluster-forming threshold
    jstat = jax.block_until_ready(jnp.asarray(stat))
    ref = np_cluster_extent(thr)(stat)  # scipy label + size (fp64) = oracle

    def _chain(s: Any) -> Any:  # fused under one jit by the harness
        return cluster_size_map(supra_threshold_clusters(s, thr))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jstat,) if framework == 'jax' else (stat,)

    baselines = {
        'nitrix-jax': ('jax', _chain),
        'fsl.fsl-cluster': ('fsl', fsl_cluster_osize(thr)),  # the bundle
        'fsl.iofloor': ('fsl', fsl_cluster_iofloor()),       # round-trip floor
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


_SHAPES = [(32, 32, 32), (48, 48, 48), (64, 64, 64)]
_LARGE = [(96, 96, 96), (128, 128, 128)]

CASE = Case(
    name='cluster_extent',
    # the chain's distinguishing op (the CC-labelling that lacks atomic
    # community parity); cluster_size_map is covered by its own atomic case.
    op_qualname='nitrix.stats.inference.cluster.supra_threshold_clusters',
    output_independent=False,  # the component a voxel joins couples the volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': list(s), 'seed': 0} for s in _SHAPES],
    representative={'shape': [48, 48, 48], 'seed': 0},
    large_param_points=tuple({'shape': list(s), 'seed': 0} for s in _LARGE),
    complexity=(
        'the cluster-extent pipeline threshold -> CC label -> per-voxel size, '
        'O(V) (the iterative label-propagation dominates). '
        'nitrix fuses the chain behind ONE jit; FSL fsl-cluster does the same '
        'bundle in one CPU pass but pays a NIfTI round-trip (subtracted via '
        'fsl.iofloor). HBM ~ V. Scale axis = the spatial volume V.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
    slow_baselines=(
        SlowBaseline(
            'fsl.fsl-cluster',
            'FSL fsl-cluster is CPU-only and file-coupled (writes the stat '
            'NIfTI, subprocess, reads the size image back); the round-trip '
            'dominates at small volumes and it has no GPU path -> declared '
            'slow, run in the full matrix.'),
    ),
    approximate_baselines=(
        ApproxBaseline(
            'fsl.iofloor',
            'no-op: returns zeros, so its rel_to_tol is large and MEANINGLESS '
            '-- the row exists only to time the NIfTI round-trip fsl-cluster '
            'pays (economic subtracts the same-namespace floor).'),
    ),
)
