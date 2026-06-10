# -*- coding: utf-8 -*-
"""Registration recipe (R3 v2): ``nitrix.register.volreg`` -- motion realign.

Batched **rigid motion realignment** of a ``(T, *spatial)`` series to a common
reference -- the ``3dvolreg`` / ``mcflirt`` task.  nitrix ``vmap``-batches all
``T`` frames on the GPU behind **one** compile (the reference work hoisted
once); the deliverable read is ``.params`` (the per-frame ``(T, p)`` rigid
transforms -- forces the full batched scan to run).  Task-level (no shared
oracle: nitrix vs the ref converge to different per-frame transforms), so
``fp64_reference`` is ``None``; the realignment is pinned in the tests (the
realigned series' inter-frame variance drops vs the raw series).

**Reference caveat (ecological validity).**  ANTsPy ``motion_correction`` is
the **available** ITK-backed reference here, but ANTs is *seldom* used for
realignment in practice -- the community standards are **AFNI ``3dvolreg``**
and **FSL ``mcflirt``** (fast, hand-optimised C), neither currently installed.
So the volreg economic verdict (``tools/economic_report.py``) is
**provisional**: a fast community tool would *shrink* any GPU win, and crowning
a multiplicative win against ANTs alone would inflate it.  ``3dvolreg`` /
``mcflirt`` are a planned ``/scratch`` install (revisit with BBR).

The economic *hypothesis* this case sets up: nitrix batches the whole series in
one compile, while ANTs realigns **frame-by-frame on CPU** (measured ~57-68
ms/frame, so ~30 s at T=500) -- so the gap should grow with ``T``.  ANTs is a
``slow_baseline`` at large ``T``.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import RegistrationSpec, volreg

from ._base import BuiltPoint, Case, SlowBaseline
from ._register import ants_motion_correction, motion_series


def _build(param: Dict[str, Any]) -> BuiltPoint:
    shape = tuple(param['shape'])
    series = motion_series(shape, int(param['T']), param.get('seed', 0))
    spec = RegistrationSpec(levels=int(param['levels']),
                            iterations=int(param['iters']))
    sj = jax.block_until_ready(jnp.asarray(series))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (sj,)
        return (series,)  # numpy series for ANTs

    baselines = {
        # return the per-frame params (the deliverable): forces the full
        # vmap-batched coarse-to-fine scan over all T frames to run.
        'nitrix-jax': ('jax', lambda s: volreg(s, spec=spec).params),
        # AVAILABLE ITK-backed moco -- NOT the community standard (see module
        # docstring: AFNI 3dvolreg / FSL mcflirt are, and are fast).
        'ants.motion_correction': ('ants', ants_motion_correction('Rigid')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level realignment: no shared oracle (nitrix vs '
                       'ANTs recover different per-frame transforms). Reads '
                       'steady + one-time compile; realignment (inter-frame '
                       'variance drops) pinned in the tests. The economic '
                       'verdict is PROVISIONAL pending AFNI 3dvolreg / FSL '
                       'mcflirt (the fast community standards).'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: vary T (the batch axis) at a small volume / cheap config, to show
# the one-compile batched cost. Representative = the cheap T=8 (for drift).
_SHAPE = [32, 32, 32]
_TDEV = [8, 16, 32]
# Size tier: the T axis is the headline (the batch / amortisation story) --
# whole-BOLD-run scale -- swept at a realistic volume; plus a volume sweep at
# fixed T. HBM ~ T*N (the realigned series + vmap working set): kept
# conservative (an OOM at the top is reported as signal, not hidden).
_LARGE = [{'shape': [48, 48, 48], 'T': t, 'levels': 2, 'iters': 20, 'seed': 0}
          for t in (50, 100, 200, 500)]
_LARGE += [{'shape': s, 'T': 100, 'levels': 2, 'iters': 20, 'seed': 0}
           for s in ([64, 64, 64], [80, 80, 80])]

CASE = Case(
    name='volreg',
    op_qualname='nitrix.register.volreg',
    output_independent=False,  # frames couple via the shared mean reference
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'T': t, 'levels': 1, 'iters': 10,
                   'seed': 0} for t in _TDEV],
    representative={'shape': _SHAPE, 'T': 8, 'levels': 1, 'iters': 10,
                    'seed': 0},
    large_param_points=tuple(_LARGE),
    slow_baselines=(
        SlowBaseline(
            'ants.motion_correction',
            reason='ANTs realigns frame-by-frame on CPU (~57-68 ms/frame '
                   'measured) -- ~14 s at T=200, ~30 s at T=500; drop in dev '
                   'cycles, worker-timeout-capped in the full matrix. (Also '
                   'not the community moco standard -- see the module note.)'),
    ),
    complexity=(
        'STEADY ~ T x iters x N per-frame, but the reference work (pyramid, '
        'inverse-compositional steepest-descent + Hessian) is hoisted once '
        'and the T frames are vmap-batched behind ONE compile -- so '
        'nitrix-GPU stays sublinear in T once the batch fills the device, '
        'while ANTs is T sequential CPU registrations (~T x 60 ms). The '
        'GPU:CPU gap should GROW with T (the batching/amortisation story), '
        'but the honest CPU bar is the FAST community tool (3dvolreg / '
        'mcflirt), '
        'not ANTs -- so the verdict is provisional. HBM ~ T*N (realigned '
        'series + vmap working set) -- the binding constraint; OOM at the '
        'top is reported as signal. Size tier varies T (headline) + volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
