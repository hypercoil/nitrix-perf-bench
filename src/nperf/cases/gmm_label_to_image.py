# -*- coding: utf-8 -*-
"""Tier-2 augmentation: ``nitrix.augment.gmm_label_to_image`` vs numpy / cupy.

SynthSeg-style generative step: render an image from a label map by per-label
Gaussian sampling -- ``out[v] = clamp(means[label[v]] + stds[label[v]]·z, 0)``.
**RNG op -- no cross-framework oracle** (`fp64_reference=None`): each framework
draws its own noise, so the ratio is a task-level wall-clock comparison and the
defining *distributional property* (per-label sample mean ≈ the label's mean)
is checked in tests. cupy is the GPU headline ref; numpy the CPU floor. No
MONAI analog (label→image GMM synthesis is SynthSeg/lab2im territory, absent
from MONAI core). Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.augment import gmm_label_to_image

from ._augment import cupy_gmm, gmm_labels, np_gmm
from ._base import BuiltPoint, Case, to_cupy


def _build(param: Dict[str, Any]) -> BuiltPoint:
    seed = int(param.get('seed', 0))
    n_labels = int(param.get('n_labels', 5))
    lab, means, stds = gmm_labels(param['shape'], n_labels, seed)
    jl = jax.block_until_ready(jnp.asarray(lab))
    jm = jnp.asarray(means)
    js = jnp.asarray(stds)
    key = jax.random.PRNGKey(seed)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(lab)
        if framework == 'jax':
            return (jl, jm, js, key)
        return (lab,)  # numpy twin closes over means / stds

    baselines = {
        'nitrix-jax': ('jax', lambda lm, m, s, k: gmm_label_to_image(
            lm, m, s, k)),
        'numpy.gmm_label_to_image': ('numpy', np_gmm(means, stds, seed)),
        'cupy.gmm_label_to_image': ('cupy', cupy_gmm(means, stds, seed)),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='RNG op (per-voxel Gaussian draw): no cross-framework '
                      'oracle; ratio is task-level wall-clock, the per-label '
                      'sample-mean ≈ label mean property is checked in tests.',
        ratio_reference='nitrix-jax',
    )


# (cube side): draw n^3 per-voxel Gaussians + a gather over per-label params.
_SIZES = [64, 96, 128]

CASE = Case(
    name='gmm_label_to_image',
    op_qualname='nitrix.augment.gmm_label_to_image',
    output_independent=True,  # each voxel is sampled independently
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': [s, s, s], 'n_labels': 5, 'seed': 0}
                  for s in _SIZES],
    representative={'shape': [96, 96, 96], 'n_labels': 5, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
