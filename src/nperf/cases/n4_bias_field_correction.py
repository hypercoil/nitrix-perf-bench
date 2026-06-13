# -*- coding: utf-8 -*-
"""Tier-2 domain-tool ref: ``nitrix.bias.n4_bias_field_correction`` vs ITK.

N4 INU / bias-field correction -- nitrix (jax) vs **SimpleITK**'s
``N4BiasFieldCorrectionImageFilter`` (the ITK/ANTs algorithm nitrix targets,
with the exact parity parameters: 4x50 iterations, 4 control points, 200
histogram bins, FWHM 0.15, Wiener noise 0.01, spline order 3). SimpleITK is the
*canonical* reference, not a mere floor.

N4's bias field is defined only up to a **global scale**, so its correctness is
a global criterion (correlation + scale-invariant RMSE over the mask), not
elementwise -- hence `fp64_reference=None`, with SimpleITK parity (corr >
0.999, scaled-relative-RMSE < 5e-3) re-asserted in `tests/test_itk_cases.py`.

SimpleITK's N4 is the slow side (~1.3-4 s/call vs nitrix's ~0.05-0.15 s run
after compile), so it is a `slow_baseline` (skipped under `--skip-slow`).
Ratio vs SimpleITK. (No GPU ref: cupy has no N4 primitive.)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.bias import n4_bias_field_correction

from ._base import BuiltPoint, Case, SlowBaseline
from ._itk import phantom, sitk_n4


def _build(param: Dict[str, Any]) -> BuiltPoint:
    s = param['s']
    obs, mask = phantom(s, param.get('seed', 7))
    jo = jax.block_until_ready(jnp.asarray(obs))
    jm = jax.block_until_ready(jnp.asarray(mask))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (obs, mask) if framework == 'numpy' else (jo, jm)

    baselines = {
        'nitrix-jax': ('jax',
                       lambda o, m: n4_bias_field_correction(o, mask=m)),
        'simpleitk.N4': ('simpleitk', sitk_n4()),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=None,
        fidelity_note="no fp64 oracle: N4's field is scale-free; SimpleITK "
                      'parity (corr > 0.999, scaled-rel-RMSE < 5e-3 over the '
                      'mask) is asserted in tests',
        ratio_reference='nitrix-jax',
    )


# cubic phantom s^3; iterative (4 fitting levels x 50 iterations).
_SIZES = [32, 48, 64]

CASE = Case(
    name='n4_bias_field_correction',
    op_qualname='nitrix.bias.n4_bias_field_correction',
    tier='marquee',
    output_independent=False,  # iterative B-spline fit couples the volume
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'s': s, 'seed': 7} for s in _SIZES],
    representative={'s': 48, 'seed': 7},
    build=_build,
    # SimpleITK's N4 is ~1.3-4 s/call (vs nitrix's ~0.05-0.15 s run); skip it
    # in fast dev cycles, pay it for the authoritative sweep.
    slow_baselines=(SlowBaseline(
        'simpleitk.N4',
        '~1.3-4 s/call on the L4 host (iterative 4x50 B-spline fit); '
        'measured 2026-06-02'),),
    rtol=5e-3,
    atol=5e-3,
)
