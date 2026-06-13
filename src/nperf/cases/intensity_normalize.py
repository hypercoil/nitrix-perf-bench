# -*- coding: utf-8 -*-
"""Tier-2 numerics: ``nitrix.numerics.intensity_normalize`` vs numpy / cupy.

Percentile-clip to ``[p1, p99]`` then rescale to ``[0, 1]`` over the whole
tensor (``axis=None``) -- the synthstrip / SynthSeg pre-training convention --
nitrix (jax) vs the numpy reimplementation (percentile clip + rescale; CPU
floor + fp64 oracle) + a CuPy GPU ref. The percentile method matches (both
linear), verified equal to 0.0 in fp64. Memory-bound, GPU-pure. Ratio vs
nitrix-jax.

**Real-anatomy point (marquee real-data input).** Beside the synthetic
matrices, a `data='mni152'` point clips+rescales the *real* MNI152 T1 (raw
positive intensities) -- the actual preprocessing step (`real_full`: no planted
truth). The numpy fp64 oracle + CuPy GPU ref carry over exactly to real data.

**Domain reference: an honest gap (no parity-grade community CLI).** Unlike
N4 / bilateral (a canonical SimpleITK engine), percentile-clip-rescale is a
numpy-level primitive with no faithful CLI competitor: ANTs
`TruncateImageIntensity` (the nearest, used in antsCorticalThickness) computes
its quantiles from a *coarse histogram*, so on the real MNI152 T1 its [0,1]
output diverges ~0.32 from the exact percentile -- a different number, not a
small tradeoff.
So the apples-to-apples bar here is the **strong CuPy GPU ref (exact)** + the
numpy fp64 oracle, not a forced domain baseline (cf. the user's principle: an
honest gap is a finding, not a baseline to manufacture).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.numerics import intensity_normalize

from ._base import BuiltPoint, Case, to_cupy
from ._normalize import cupy_intensity, normalize_input, np_intensity
from ._real_anatomy import load_mni152_raw


def _build(param: Dict[str, Any]) -> BuiltPoint:
    if param.get('data') == 'mni152':
        X = load_mni152_raw(int(param.get('resolution', 2)))  # real anatomy
    else:
        X = normalize_input(param['n'], param.get('seed', 0))
    jx = jax.block_until_ready(jnp.asarray(X))
    ref = np_intensity(X.astype('float64'))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: intensity_normalize(x)),
        'numpy.intensity': ('numpy', np_intensity),  # CPU floor
        'cupy.intensity_normalize': ('cupy', cupy_intensity()),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (rows = cols): percentiles over the whole tensor; cost ~ n² (+ a sort).
_SIZES = [512, 2048, 4096]
# Real-anatomy point: the raw MNI152 T1 (~1.1M voxels @2mm) -- the marquee
# real-data input (the actual percentile-norm preprocessing step). real_full:
# no planted truth; validated by the same numpy fp64 oracle + CuPy GPU ref.
_REAL = {'data': 'mni152', 'resolution': 2, 'realism': 'real_full'}

CASE = Case(
    name='intensity_normalize',
    op_qualname='nitrix.numerics.intensity_normalize',
    tier='marquee',
    output_independent=False,  # global percentiles couple every element
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _SIZES] + [_REAL],
    representative={'n': 2048, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
