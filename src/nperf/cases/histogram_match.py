# -*- coding: utf-8 -*-
"""Tier-2 domain-tool ref: ``nitrix.bias.histogram_match`` vs SimpleITK.

Nyul-Udupa landmark histogram matching / standardisation -- nitrix (jax) vs
**SimpleITK**'s ``HistogramMatchingImageFilter`` (the ITK algorithm nitrix
targets: 1024 histogram levels, 7 match points, threshold-at-mean). SimpleITK
is the *canonical* reference here, not a mere floor: nitrix's own suite asserts
live-ITK parity, and this case re-asserts it (max diff < 1e-3 of the reference
range) as the correctness gate in place of an fp64 oracle.

No bit-level fp64 oracle (`fp64_reference=None`): the truth *is* the ITK
landmark map, so parity-with-SimpleITK is the meaningful correctness check
(asserted in `tests/test_itk_cases.py`). Pure sort/gather/interp, so GPU-pure.
Ratio vs SimpleITK. (No GPU ref: cupy has no histogram-matching primitive.)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.bias import histogram_match

from ._base import BuiltPoint, Case
from ._itk import sitk_histogram_match, synth_pair


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = param['n']
    src, ref = synth_pair(n, param.get('seed', 11))
    js = jax.block_until_ready(jnp.asarray(src))
    jr = jax.block_until_ready(jnp.asarray(ref))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (src, ref) if framework == 'numpy' else (js, jr)

    baselines = {
        'nitrix-jax': ('jax', lambda s, r: histogram_match(s, r)),
        'simpleitk.HistogramMatching': ('simpleitk', sitk_histogram_match()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note='no fp64 oracle: truth is ITK landmark map; '
                      'SimpleITK parity (max diff < 1e-3 of ref range) is '
                      'asserted in tests',
        ratio_reference='nitrix-jax',
    )


# source n^3 (reference (n+8)^3); cost ~ O(voxels) + a sort for landmarks.
_SIZES = [32, 64, 96]

CASE = Case(
    name='histogram_match',
    op_qualname='nitrix.bias.histogram_match',
    output_independent=False,  # each voxel maps through shared landmarks
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 11} for n in _SIZES],
    representative={'n': 64, 'seed': 11},
    build=_build,
    rtol=1e-3,
    atol=1e-3,
)
