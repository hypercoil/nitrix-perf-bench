# -*- coding: utf-8 -*-
"""Registration recipe (R2 v2): ``nitrix.register.greedy_syn_register``.

Greedy **symmetric diffeomorphic** registration (SyN-style): symmetric
forward/inverse velocity fields driven to a midpoint by a local-NCC force, the
canonical deformable recipe.  Task-level (no shared oracle: nitrix LNCC, ANTs'
CC, and dipy's SSD converge to *different* warps), so ``fp64_reference`` is
``None`` and the diffeomorphic recovery (``ncc`` up, ``jacobian_det > 0``) is
pinned in ``tests/test_register_cases.py``.

The headline read is nitrix **steady** (warm) + its **one-time compile** vs the
gold standard's full wall-clock.  Two CPU domain references (own refs envs, not
jit-compiled): **ANTsPy** ``registration(SyNOnly)`` -- the canonical SyN, run
**deformable-only** to match nitrix's greedy SyN (no affine pre-step) -- and
**dipy** ``SymmetricDiffeomorphic`` (SSD).  Because even this deformable gold
standard is only *seconds*-to-minutes on CPU, the GPU win must be **earned** --
it has to clear the ~4x hardware-cost bar to count, which the economic verdict
measures rather than assumes (see ``tools/economic_report.py``).  Ratio vs
``nitrix-jax``.

**Matched schedule (the size / economic tier).**  ANTs is treated as
**canonical** and all three tools run the SAME pyramid there: 3 levels (shrink
``4x2x1``), per-level iters ``(40, 20, 10)``, with ANTs' ``[...,1e-7,8]``
early-exit mirrored on nitrix via ``Convergence`` and on dipy via matched
``level_iters`` -- so the wall-clock is apples-to-apples (no
different-work confound; matched-schedule fairness).  NB ANTs' *default*
``SyNOnly`` preset is ``(40, 20, 0)`` which **skips the full-res level** (0
iters there -- measured ~0.5 / 2.9 / 6.0 s at 48 / 96 / 128^3); the matched
``(40, 20, 10)`` makes it do real full-res work (~3.3x slower, ~15 s at 96^3
measured) -- the honest, heavier bar.  dipy is the genuinely slow tool at this
schedule (full-res cython, super-linear) -- the lone ``slow_baseline``.  (The
dev tier keeps a flat ``(levels, iters)`` for compile characterisation, where
the verdict is *not* read -- see the economic-report caveats.)

**The driving force is a benchmarked knob (fMRIPrep parity).**  The default
``nitrix-jax`` uses the mono-modal LNCC force; two extra nitrix baselines swap
in the **mutual-information** force that fMRIPrep's SyN-SDC / cross-modal
deformable registration uses: ``nitrix-jax-mi`` is the closed-form **Mattes
MI** force (``MIForce`` -- ``mi_grad`` + a joint-histogram scatter, the fast
cross-modal path), and ``nitrix-jax-mi-autodiff`` is the generic autodiff
``MetricForce(MI())`` (the same metric driven by ``jax.grad`` of its soft-
histogram cost).  The two MI rows are *both* the **closed-form-vs-autodiff
speed lever** (does the hand-written Tier-1 force earn its keep? -- measured
per platform, not assumed: on CPU the closed form is *not* automatically
faster, a finding the matrix surfaces) and a **parity oracle** (agree in force
direction -- nitrix's S3 oracle).  ANTs ``SyNOnly`` already defaults to the
**mattes-MI** metric (32 bins), so it doubles as the apples-to-apples community
bar for the MI variants (the fMRIPrep tool itself).  The MI ranges are pinned
once from the full-res pair (a non-stationary objective).  Note these pairs are
mono-modal (MI's cost is modality-independent, so the *perf* read is faithful);
a genuinely cross-modal (T1<->T2 / EPI) input pair is a tracked input-realism
follow-up.

The size tier also carries **anisotropic** points (``spacing=[1,1,3]``): the op
corrects the bias where a voxel-isotropic Gaussian / force is physically
anisotropic (``SyNSpec.spacing``); the refs get the matching voxel->world
affine so the comparison stays in physical space.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.register import (
    MI,
    Convergence,
    MetricForce,
    MIForce,
    SyNSpec,
    greedy_syn_register,
)

from ._base import BuiltPoint, Case, SlowBaseline
from ._real_anatomy import real_syn_pair
from ._register import (
    _affine,
    aniso_pair,
    ants_register,
    dipy_register,
    syn_pair,
)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    # A scalar 'iters' is the legacy flat schedule (dev tier, no early-exit);
    # a per-level tuple (e.g. [40, 20, 10]) is the ANTs-canonical MATCHED
    # schedule -- 3 levels (shrink 4x2x1 via pyramid_factor=2), with ANTs'
    # [...,1e-7,8] early-exit mirrored on nitrix (Convergence) so the economic
    # wall-clock is apples-to-apples (the size/economic tier; see the module
    # note + ants_register/_dipy_pyramid). ants_iters/dipy get the same tuple.
    raw = param['iters']
    if isinstance(raw, (list, tuple)):
        iters: Any = tuple(int(i) for i in raw)
        levels = len(iters)
        conv = Convergence(threshold=1e-7, window=8)
        ants_iters: Any = iters
    else:
        iters = int(raw)
        levels = int(param['levels'])
        conv = None
        ants_iters = None
    spacing = param.get('spacing')  # None -> isotropic (voxel space)
    seed = param.get('seed', 0)
    if param.get('data') == 'mni152':
        # REAL anatomy: MNI152 T1 under a smooth non-rigid warp (a small
        # background noise floor; see _real_anatomy).
        moving, fixed = real_syn_pair(int(param.get('resolution', 2)), seed)
        spec = SyNSpec(levels=levels, iterations=iters, convergence=conv)
        ants_ref = ants_register('SyNOnly', reg_iterations=ants_iters)
        dipy_ref = dipy_register('syn', levels, iters)
    elif spacing is not None:
        moving, fixed, sp = aniso_pair(tuple(param['shape']), spacing, seed)
        spec = SyNSpec(levels=levels, iterations=iters, spacing=sp,
                       convergence=conv)
        aff = _affine(spacing)
        ants_ref = ants_register('SyNOnly', spacing=list(spacing),
                                 reg_iterations=ants_iters)
        dipy_ref = dipy_register('syn', levels, iters, affines=(aff, aff))
    else:
        moving, fixed = syn_pair(tuple(param['shape']), seed)
        spec = SyNSpec(levels=levels, iterations=iters, convergence=conv)
        ants_ref = ants_register('SyNOnly', reg_iterations=ants_iters)
        dipy_ref = dipy_register('syn', levels, iters)

    mj = jax.block_until_ready(jnp.asarray(moving))
    fj = jax.block_until_ready(jnp.asarray(fixed))
    # MI force ranges, pinned ONCE from the (host) full-res pair -- a data
    # min/max range drifts as the moving image deforms (a non-stationary
    # objective), and under jax.jit the recipe cannot resolve a None range from
    # traced images (float(tracer) is not eager). Same 32 bins as ANTs mattes.
    rng_m = (float(moving.min()), float(moving.max()))
    rng_f = (float(fixed.min()), float(fixed.max()))
    mi_force = MIForce(bins=32, range_moving=rng_m, range_fixed=rng_f)
    mi_metric = MetricForce(MI(bins=32, range_moving=rng_m, range_fixed=rng_f))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'jax':
            return (mj, fj)
        return (moving, fixed)  # numpy for ANTs / dipy

    baselines = {
        # return the displacement field (the deliverable): forces the full
        # forward+inverse integrate + midpoint compose to run (real compile).
        # DEFAULT force = LNCCForce (the mono-modal local-CC path).
        'nitrix-jax': (
            'jax', lambda mv, fx: greedy_syn_register(mv, fx, spec=spec
                                                      ).displacement),
        # fMRIPrep's metric: the closed-form Mattes-MI force (the cross-modal
        # deformable fast path -- mi_grad + histogram scatter, no autodiff
        # tape). Same recipe + a force knob, so the ratio vs nitrix-jax is the
        # MI-vs-LNCC force cost on the SAME pair.
        'nitrix-jax-mi': (
            'jax', lambda mv, fx: greedy_syn_register(
                mv, fx, spec=spec, force=mi_force).displacement),
        # the autodiff escape hatch MetricForce(MI()): the parity oracle for
        # the closed form (agrees in direction/magnitude -- nitrix S3) and
        # the closed-form-vs-autodiff SPEED LEVER (is the Tier-1 hand-written
        # force worth it? -- measured per platform, not assumed).
        'nitrix-jax-mi-autodiff': (
            'jax', lambda mv, fx: greedy_syn_register(
                mv, fx, spec=spec, force=mi_metric).displacement),
        # ANTs SyNOnly defaults to the mattes-MI metric (32 bins) -- so it is
        # the apples-to-apples community bar for the MI variants (the fMRIPrep
        # registration tool itself), as well as the SyN gold standard.
        'ants.registration': ('ants', ants_ref),  # gold std (fast: SyNOnly)
        'dipy.registration': ('dipy', dipy_ref),   # dipy diffeo (the slow one)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level deformable recipe: no shared oracle '
                       '(nitrix LNCC vs ANTs CC vs dipy SSD converge to '
                       'different warps). Reads steady + one-time compile; '
                       'recovery (ncc up, jacobian_det>0) pinned in tests.'),
        ratio_reference='nitrix-jax',
    )


# Dev tier: configs at one small volume, varying (levels, iterations) to show
# the per-level lax.scan keeps compile ~flat in iters. Representative = the
# cheap L1x40 (for --quick / drift).
_CONFIGS = [(1, 40), (2, 80), (3, 80)]
_SHAPE = [48, 48, 48]
# Size tier (brain-scale): vary the volume at the ANTs-CANONICAL matched
# schedule -- 3 levels (shrink 4x2x1), per-level iters (40, 20, 10), early-exit
# 1e-7/win-8 -- so the economic verdict reads apples-to-apples vs ANTs / dipy
# (the per-level tuple triggers the matched path in _build; ANTs gets the same
# reg_iterations, dipy the same level_iters). Plus anisotropic (1x1x3) points.
_MATCHED = [40, 20, 10]
_LARGE = [{'shape': s, 'levels': 3, 'iters': _MATCHED, 'seed': 0}
          for s in ([64, 64, 64], [96, 96, 96], [128, 128, 128])]
_LARGE += [{'shape': s, 'levels': 3, 'iters': _MATCHED, 'seed': 0,
            'spacing': [1, 1, 3]} for s in ([64, 64, 64], [96, 96, 96])]
# Real-anatomy point: MNI152 T1 (~99^3 @2mm) under a smooth non-rigid warp.
_LARGE += [{'data': 'mni152', 'resolution': 2, 'levels': 3,
            'iters': _MATCHED, 'seed': 0}]

CASE = Case(
    name='greedy_syn_register',
    op_qualname='nitrix.register.greedy_syn_register',
    tier='marquee',
    output_independent=False,  # a global iterative deformable fit
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'shape': _SHAPE, 'levels': lv, 'iters': it, 'seed': 0}
                  for (lv, it) in _CONFIGS],
    representative={'shape': _SHAPE, 'levels': 1, 'iters': 40, 'seed': 0},
    large_param_points=tuple(_LARGE),
    # Only dipy is slow: ANTs SyNOnly is FAST (~6 s at 128^3, measured), so it
    # stays in dev cycles as the gold-standard bar; dipy SyN is the slow one.
    slow_baselines=(
        SlowBaseline('dipy.registration',
                     reason='dipy SymmetricDiffeomorphic, super-linear cython '
                            'on CPU; at the matched (40,20,10) full-res '
                            'schedule it is the slow tool -- worker-timeout-'
                            'capped in the full matrix. ANTs SyNOnly runs the '
                            'same schedule (~15 s at 96^3) but completes.'),
    ),
    complexity=(
        'STEADY ~ levels x iters x n_steps x N: each iteration warps both '
        'images to the midpoint (two scaling-and-squaring SVF integrations), '
        'computes the LNCC force, smooths it (fluid) + the velocity '
        '(diffusion) -- two Gaussians/iter -- then a midpoint compose+invert '
        'at the end. The heaviest recipe to COMPILE (two velocity fields). '
        'The size tier runs the ANTs-CANONICAL matched schedule (3 levels, '
        '40x20x10, early-exit 1e-7/8) on all three tools -- ANTs at its '
        'DEFAULT skip-full-res preset was ~0.5/2.9/6.0 s at 48/96/128^3, but '
        'the matched full-res schedule is ~3.3x heavier (~15 s at 96^3, the '
        'honest apples-to-apples bar); the GPU win must still clear ~4x to '
        'count (measured in ECONOMIC.md, not assumed). HBM ~ 2 velocity '
        'fields + scaling-squaring intermediates (heaviest after demons). The '
        'size tier varies the volume + carries anisotropic (1x1x3) points. '
        'The force is a benchmarked knob: '
        'nitrix-jax (LNCC) vs the MI force (closed-form MIForce + autodiff '
        'MetricForce(MI)) -- MI replaces the local-CC window with a '
        'joint-histogram scatter (modality-independent cost; fMRIPrep parity, '
        'ANTs SyNOnly = mattes MI is the bar).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
