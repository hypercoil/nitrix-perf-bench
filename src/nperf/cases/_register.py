# -*- coding: utf-8 -*-
"""Shared helpers for the registration *recipe* cases (R1-R3 drivers).

Unlike the op-vs-oracle cases, a registration recipe is an **end-to-end,
iterative** driver: different tools (nitrix GN/LM, ANTs, dipy) converge to
*different* transforms, so there is **no shared cross-impl oracle**.  The case
is therefore **task-level**: plant a known warp, register, and read

- the **time split** -- ``compile_time`` (the cold first call) vs
  ``steady_time`` (warm).  This is the headline: nitrix's recipes are
  Python-unrolled fixed-iteration loops, so the *cold compile* scales with the
  total unrolled iteration count and dominates first-call latency (the
  "registration slow on GPU" diagnosis; see the filed nitrix finding
  ``registration-recipe-cold-compile``).  The suite already separates
  compile from steady, so the cold compile is a first-class measured number,
  not hidden inside "slow".
- the **recovery accuracy** (does the recipe recover the planted warp), pinned
  in ``tests/test_register_cases.py`` (not gated in the bench:
  ``fp64_reference`` is ``None`` -- no oracle).

The planted warp is applied with **scipy** (independent of nitrix's own
``spatial_transform``), so the input is not produced by the op family under
test.  **Two** task-level domain references run alongside nitrix, both CPU and
neither jit-compiled (so each one's wall-clock is the full registration with no
separate compile -- the honest framing is nitrix *steady* (post-compile) + its
one-time compile, vs the tool's wall-clock):

- ANTsPy (``ants_register`` -> ``ants.registration``): ITK-backed C++, runs its
  own fixed internal multi-resolution schedule (it ignores our ``levels`` /
  ``iterations``), so its wall-clock is roughly flat across our param points.
- **dipy** (``dipy_register``): numpy / scipy / cython.  Its pyramid is
  settable, so -- unlike ANTs -- we drive it with the *same* ``(levels,
  iterations)`` the nitrix recipe uses, and its wall-clock scales with that
  budget like nitrix's steady does (the apples-to-apples per-config foil).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np


def warp_pair(shape: Sequence[int], seed: int = 0
              ) -> Tuple[np.ndarray, np.ndarray]:
    '''A ``(moving, fixed)`` pair: ``fixed`` is structured (smoothed noise),
    ``moving`` is ``fixed`` under a small known **rigid** warp applied with
    scipy (independent of nitrix), so a correct recipe recovers it.'''
    import scipy.ndimage as spnd
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    fixed = spnd.gaussian_filter(
        rng.standard_normal(tuple(shape)).astype(np.float32), 2.0)
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-6)
    rot = Rotation.from_euler('xyz', [3.0, 2.0, 1.5], degrees=True).as_matrix()
    center = (np.asarray(shape) - 1) / 2.0
    offset = center - rot @ center + np.array([1.5, -1.0, 0.8])
    moving = spnd.affine_transform(fixed, rot, offset=offset, order=1,
                                   mode='nearest')
    return moving.astype(np.float32), fixed.astype(np.float32)


def ncc(a: Any, b: Any) -> float:
    '''Global Pearson correlation -- the recovery-quality read (warped vs
    fixed): higher means the recipe registered better.'''
    x = np.asarray(a, np.float64).ravel()
    y = np.asarray(b, np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum()) + 1e-8
    return float((x * y).sum() / den)


def ants_register(transform: str = 'Rigid') -> Callable[..., Any]:
    '''ANTsPy registration (the task-level domain reference) for the given
    ``type_of_transform`` (``'Rigid'`` / ``'Affine'`` / ``'SyN'`` -- the
    rigid / affine / diffeomorphic counterparts of the nitrix recipes);
    returns the warped moving.  ants lazy (only its refs env imports it); not
    jit-compiled, so its wall-clock is the full registration (no separate
    compile) -- read against nitrix's steady + one-time compile.'''

    def run(moving: Any, fixed: Any) -> Any:
        import ants

        f = ants.from_numpy(np.asarray(fixed, np.float32))
        m = ants.from_numpy(np.asarray(moving, np.float32))
        reg = ants.registration(fixed=f, moving=m,
                                type_of_transform=transform)
        return reg['warpedmovout'].numpy()

    return run


def _dipy_pyramid(levels: int, iters: int
                  ) -> Tuple[list, list, list]:
    '''The coarse->fine schedule for dipy, truncated to ``levels`` stages and
    ``iters`` iterations per stage -- the same knob the nitrix recipe uses, so
    dipy's work tracks the config (the standard downsample factors / smoothing
    sigmas, finest ``levels`` of ``[4,2,1]`` / ``[3,1,0]``).'''
    factors = [4, 2, 1][-levels:]
    sigmas = [3.0, 1.0, 0.0][-levels:]
    level_iters = [iters] * levels
    return level_iters, sigmas, factors


def dipy_register(kind: str, levels: int, iters: int) -> Callable[..., Any]:
    '''dipy registration (the numpy/scipy/cython task-level domain reference),
    returning the warped moving.  ``kind`` picks the counterpart of the nitrix
    recipe: ``'rigid'`` / ``'affine'`` -> ``AffineRegistration`` (mutual
    information, the rigid / 12-DOF transforms), ``'syn'`` -> the symmetric
    diffeomorphic ``SymmetricDiffeomorphicRegistration`` on SSD (the
    counterpart of nitrix's SSD-driven log-Demons).  Its pyramid is driven by
    the case's ``(levels, iters)`` via ``_dipy_pyramid``, so dipy's wall-clock
    scales with the same budget.  dipy lazy (only its refs env imports it); not
    jit-compiled, so its wall-clock is the full registration (no separate
    compile) -- read against nitrix's steady + one-time compile.'''

    def run(moving: Any, fixed: Any) -> Any:
        # static = fixed, moving = moving; register moving->fixed and apply to
        # moving (recovery == ncc(warped, fixed)).  dipy works in fp64.
        stat = np.asarray(fixed, np.float64)
        mov = np.asarray(moving, np.float64)
        level_iters, sigmas, factors = _dipy_pyramid(levels, iters)
        if kind == 'syn':
            from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
            from dipy.align.metrics import SSDMetric

            sdr = SymmetricDiffeomorphicRegistration(
                SSDMetric(3), level_iters=level_iters)
            sdr.verbosity = 0
            mapping = sdr.optimize(stat, mov)
            return mapping.transform(mov)
        from dipy.align.imaffine import (
            AffineRegistration,
            MutualInformationMetric,
        )
        from dipy.align.transforms import AffineTransform3D, RigidTransform3D

        transform = (RigidTransform3D() if kind == 'rigid'
                     else AffineTransform3D())
        areg = AffineRegistration(
            metric=MutualInformationMetric(nbins=32, sampling_proportion=None),
            level_iters=level_iters, sigmas=sigmas, factors=factors,
            verbosity=0)
        amap = areg.optimize(stat, mov, transform, None)
        return amap.transform(mov)

    return run


def sitk_demons_register(iters: int, *, sigma: float = 1.0,
                         hist_match: bool = True) -> Callable[..., Any]:
    '''SimpleITK ``DiffeomorphicDemonsRegistrationFilter`` -- the *direct*
    canonical ITK diffeomorphic-demons reference (Vercauteren), the closest
    cross-tool counterpart of nitrix's log-Demons.  Like nitrix (and unlike
    ANTs / dipy, which terminate on convergence) ITK's demons filter runs to
    its configured iteration count -- no metric-based early-exit in the basic
    filter -- so it is the fairest *per-iteration* demons comparison (see the
    fixed-iteration vs early-stop caveat in reports/REGISTRATION_SCALING.md).
    Single-resolution at the case's ``iters``, with the canonical
    histogram-match pre-step (demons assumes intensity correspondence).
    SimpleITK is lazy (base env, numpy fw); returns the warped moving.'''

    def run(moving: Any, fixed: Any) -> Any:
        import SimpleITK as sitk

        f = sitk.GetImageFromArray(np.asarray(fixed, np.float32))
        m = sitk.GetImageFromArray(np.asarray(moving, np.float32))
        if hist_match:
            hm = sitk.HistogramMatchingImageFilter()
            hm.SetNumberOfHistogramLevels(128)
            hm.SetNumberOfMatchPoints(10)
            hm.ThresholdAtMeanIntensityOn()
            m = hm.Execute(m, f)
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(int(iters))
        demons.SetStandardDeviations(sigma)
        disp = demons.Execute(f, m)
        tx = sitk.DisplacementFieldTransform(disp)
        warped = sitk.Resample(m, f, tx, sitk.sitkLinear, 0.0)
        return sitk.GetArrayFromImage(warped)

    return run
