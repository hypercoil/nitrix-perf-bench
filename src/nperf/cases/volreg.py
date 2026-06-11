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

**References (ecological validity).**  The **community** realignment standards
are **AFNI ``3dvolreg``** and **FSL ``mcflirt``** (fast, hand-optimised C) --
now installed on ``/scratch`` (``tools/setup_neuro_refs.sh``) and wired as the
``afni`` / ``fsl`` providers (binaries at ``NPERF_AFNI_DIR`` /
``NPERF_FSL_DIR``; NIfTI round-trip; realign to the mean, matching
``reference='mean'``).  ANTsPy ``motion_correction`` is kept as a secondary
ITK-backed ref but is **seldom** used for moco in practice.  The economic
verdict (``tools/economic_report.py``) picks the **fastest** CPU domain tool
as the gold standard -- so 3dvolreg / mcflirt (not the slower ANTs) set the
honest bar, which *shrinks* any GPU win vs the naive ANTs-only comparison (the
no-inflated-win discipline).

**I/O floor (harness artifact).**  The CLI tools' wall-clock includes a NIfTI
write + subprocess launch + read that nitrix (in-memory) does not pay -- pure
harness overhead.  ``afni.iofloor`` (``3dcalc -expr a``) and ``fsl.iofloor``
(``fslmaths -mul 1``) are **no-ops** with the *same* round-trip, so the
economic report subtracts them (``compute = tool - iofloor``) to isolate the
registration compute for a fair comparison vs nitrix (measured ~42% I/O at
T=50/48^3).

The economic *story*: nitrix batches the whole series in one compile, while the
CPU tools realign **frame-by-frame** -- so the gap should grow with ``T``. ANTs
(~57-68 ms/frame) is a ``slow_baseline`` at large ``T``; 3dvolreg / mcflirt are
fast and stay in dev cycles. Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import RegistrationSpec, volreg

from ._base import BuiltPoint, Case, SlowBaseline
from ._register import (
    afni_iofloor,
    afni_volreg,
    ants_motion_correction,
    fsl_iofloor,
    fsl_mcflirt,
    motion_series,
)


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
        # the COMMUNITY realignment standards (fast C; the honest CPU bar).
        'afni.3dvolreg': ('afni', afni_volreg()),
        'fsl.mcflirt': ('fsl', fsl_mcflirt()),
        # I/O-floor no-ops (3dcalc / fslmaths identity): the NIfTI round-trip
        # the economic report subtracts to isolate the registration compute.
        'afni.iofloor': ('afni', afni_iofloor()),
        'fsl.iofloor': ('fsl', fsl_iofloor()),
        # secondary ITK-backed moco -- seldom the realignment tool in practice.
        'ants.motion_correction': ('ants', ants_motion_correction('Rigid')),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level realignment: no shared oracle (nitrix vs '
                       'ANTs recover different per-frame transforms). Reads '
                       'steady + one-time compile; realignment (inter-frame '
                       'variance drops) pinned in the tests. The economic '
                       'verdict bars against AFNI 3dvolreg / FSL mcflirt (the '
                       'fast community standards), I/O-floor-subtracted.'),
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
        'GPU:CPU gap should GROW with T (the batching/amortisation story); '
        'the honest CPU bar is the FAST community tool (3dvolreg / mcflirt), '
        'I/O-floor-subtracted, not the slower ANTs (timed out at T=500). '
        'HBM ~ T*N (realigned series + vmap working set) -- the binding '
        'constraint; OOM at the top is reported as signal. Size tier varies T '
        '(headline) + volume.'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
