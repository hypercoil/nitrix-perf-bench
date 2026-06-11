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


def ants_register(transform: str = 'Rigid',
                  spacing: Any = None) -> Callable[..., Any]:
    '''ANTsPy registration (the task-level domain reference) for the given
    ``type_of_transform`` (``'Rigid'`` / ``'Affine'`` / ``'SyN'`` -- the
    rigid / affine / diffeomorphic counterparts of the nitrix recipes);
    returns the warped moving.  ants lazy (only its refs env imports it); not
    jit-compiled, so its wall-clock is the full registration (no separate
    compile) -- read against nitrix's steady + one-time compile.

    ``spacing`` (per-axis voxel size, or ``(fixed_spacing, moving_spacing)``
    for a cross-grid pair) sets the images' physical spacing so ANTs registers
    in the *same* physical space nitrix's ``WorldSpace`` uses -- the
    apples-to-apples comparison on anisotropic / different grids (ANTs is a
    physical-space tool, so this is its native regime). ``None`` keeps ANTs'
    default 1 mm isotropic (the shared-grid IndexSpace case).'''

    def run(moving: Any, fixed: Any) -> Any:
        import ants

        f = ants.from_numpy(np.asarray(fixed, np.float32))
        m = ants.from_numpy(np.asarray(moving, np.float32))
        if spacing is not None:
            sp = tuple(spacing)
            # one shared spacing, or an explicit (fixed, moving) pair.
            f_sp, m_sp = (sp, sp) if np.isscalar(sp[0]) else (sp[0], sp[1])
            f.set_spacing(tuple(float(s) for s in f_sp))
            m.set_spacing(tuple(float(s) for s in m_sp))
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


def dipy_register(kind: str, levels: int, iters: int,
                  affines: Any = None) -> Callable[..., Any]:
    '''dipy registration (the numpy/scipy/cython task-level domain reference),
    returning the warped moving.  ``kind`` picks the counterpart of the nitrix
    recipe: ``'rigid'`` / ``'affine'`` -> ``AffineRegistration`` (mutual
    information, the rigid / 12-DOF transforms), ``'syn'`` -> the symmetric
    diffeomorphic ``SymmetricDiffeomorphicRegistration`` on SSD (the
    counterpart of nitrix's SSD-driven log-Demons).  Its pyramid is driven by
    the case's ``(levels, iters)`` via ``_dipy_pyramid``, so dipy's wall-clock
    scales with the same budget.  dipy lazy (only its refs env imports it); not
    jit-compiled, so its wall-clock is the full registration (no separate
    compile) -- read against nitrix's steady + one-time compile.

    ``affines`` = ``(static_grid2world, moving_grid2world)`` voxel->world
    homogeneous affines passed to dipy's ``optimize`` so it registers in
    physical space (anisotropy- / cross-grid-correct), matching nitrix's
    ``WorldSpace``; ``None`` is voxel space (the shared-grid case).'''
    s2w, m2w = (None, None) if affines is None else affines

    def run(moving: Any, fixed: Any) -> Any:
        # static = fixed, moving = moving; register moving->fixed and apply to
        # moving (recovery == ncc(warped, fixed)).  dipy works in fp64.
        stat = np.asarray(fixed, np.float64)
        mov = np.asarray(moving, np.float64)
        sw = None if s2w is None else np.asarray(s2w, np.float64)
        mw = None if m2w is None else np.asarray(m2w, np.float64)
        level_iters, sigmas, factors = _dipy_pyramid(levels, iters)
        if kind == 'syn':
            from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
            from dipy.align.metrics import SSDMetric

            sdr = SymmetricDiffeomorphicRegistration(
                SSDMetric(3), level_iters=level_iters)
            sdr.verbosity = 0
            mapping = sdr.optimize(stat, mov, static_grid2world=sw,
                                   moving_grid2world=mw)
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
        amap = areg.optimize(stat, mov, transform, None,
                             static_grid2world=sw, moving_grid2world=mw)
        return amap.transform(mov)

    return run


def sitk_demons_register(iters: int, *, sigma: float = 1.0,
                         hist_match: bool = True, spacing: Any = None
                         ) -> Callable[..., Any]:
    '''SimpleITK ``DiffeomorphicDemonsRegistrationFilter`` -- the *direct*
    canonical ITK diffeomorphic-demons reference (Vercauteren), the closest
    cross-tool counterpart of nitrix's log-Demons.  Like nitrix (and unlike
    ANTs / dipy, which terminate on convergence) ITK's demons filter runs to
    its configured iteration count -- no metric-based early-exit in the basic
    filter -- so it is the fairest *per-iteration* demons comparison (see the
    fixed-iteration vs early-stop caveat in reports/REGISTRATION_SCALING.md).
    Single-resolution at the case's ``iters``, with the canonical
    histogram-match pre-step (demons assumes intensity correspondence).
    ``spacing`` (per-axis voxel size) sets the images' physical spacing so the
    filter corrects anisotropy in the same physical space as nitrix's
    ``DemonsSpec.spacing``; ``None`` is isotropic.  SimpleITK is lazy (base
    env, numpy fw); returns the warped moving.'''

    def run(moving: Any, fixed: Any) -> Any:
        import SimpleITK as sitk

        f = sitk.GetImageFromArray(np.asarray(fixed, np.float32))
        m = sitk.GetImageFromArray(np.asarray(moving, np.float32))
        if spacing is not None:
            # SetSpacing is x,y,z; numpy/GetImageFromArray axes are reversed.
            sp = tuple(float(s) for s in reversed(tuple(spacing)))
            f.SetSpacing(sp)
            m.SetSpacing(sp)
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


# -- cross-grid / anisotropic / non-rigid input generators -------------------


def _affine(spacing: Sequence[float]) -> np.ndarray:
    '''Voxel->world homogeneous affine for a diagonal (anisotropic) spacing --
    ``diag([*spacing, 1])`` (the NIfTI-sform contract nitrix ``WorldSpace`` and
    the ANTs/dipy refs both consume).'''
    return np.diag([*[float(s) for s in spacing], 1.0]).astype(np.float32)


def warp_pair_cross_grid(
    fixed_shape: Sequence[int],
    moving_shape: Sequence[int],
    *,
    fixed_spacing: Sequence[float] = (1.0, 1.0, 1.0),
    moving_spacing: Sequence[float] = (1.2, 1.0, 0.9),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''A cross-grid ``(moving, fixed, moving_affine, fixed_affine)``: the two
    images live on **different grids** (different shape *and* anisotropic voxel
    spacing) with a known **world-space** rigid transform between them -- the
    realistic cross-resolution / cross-modal regime (e.g. a 2 mm EPI against a
    1 mm T1) that ``IndexSpace`` (shared grid) cannot express.

    ``fixed`` is the structured ``warp_pair`` field; ``moving`` is ``fixed``
    resampled into the ``moving_shape`` grid through the composed voxel map
    ``A_moving^-1 . T_world . A_fixed`` with **scipy** (independent of nitrix),
    so the input is not produced by the op under test.  The recipe (with
    ``WorldSpace(fixed_affine, moving_affine)``) must recover ``T_world``.  The
    rotation is kept within the recipes' capture range (~3 deg, like
    ``warp_pair``).'''
    import scipy.ndimage as spnd
    from scipy.spatial.transform import Rotation

    fixed_shape = tuple(int(s) for s in fixed_shape)
    moving_shape = tuple(int(s) for s in moving_shape)
    ndim = len(fixed_shape)
    rng = np.random.default_rng(seed)
    fixed = spnd.gaussian_filter(
        rng.standard_normal(fixed_shape).astype(np.float32), 2.0)
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-6)

    a_f = _affine(fixed_spacing).astype(np.float64)
    a_m = _affine(moving_spacing).astype(np.float64)
    rot = Rotation.from_euler('xyz', [3.0, 2.0, 1.5], degrees=True).as_matrix()
    # T_world: rotate about the fixed image's world centre + a small mm shift.
    fc = a_f[:ndim, :ndim] @ ((np.asarray(fixed_shape) - 1) / 2.0)
    t_world = np.eye(ndim + 1)
    t_world[:ndim, :ndim] = rot
    t_world[:ndim, ndim] = fc - rot @ fc + np.array([1.5, -1.0, 0.8])[:ndim]
    # fixed-voxel -> moving-voxel; invert to pull fixed into the moving grid.
    m = np.linalg.inv(a_m) @ t_world @ a_f
    m_inv = np.linalg.inv(m)
    moving = spnd.affine_transform(
        fixed, m_inv[:ndim, :ndim], offset=m_inv[:ndim, ndim],
        output_shape=moving_shape, order=1, mode='nearest')
    return (moving.astype(np.float32), fixed.astype(np.float32),
            a_m.astype(np.float32), a_f.astype(np.float32))


def syn_pair(shape: Sequence[int], seed: int = 0, *,
             max_disp: float = 3.0, smooth: float = 4.0
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''An identical-shape ``(moving, fixed)`` pair related by a **smooth
    non-rigid** deformation -- the canonical diffeomorphic-recovery test for
    greedy SyN / demons (a rigid warp would understate the deformable work).
    ``fixed`` is the ``warp_pair`` field; ``moving`` is ``fixed`` pushed by a
    low-frequency random displacement (gaussian-smoothed, scaled to
    ``max_disp`` voxels) applied with **scipy** ``map_coordinates`` --
    independent of nitrix's own integrate / spatial_transform path.'''
    import scipy.ndimage as spnd

    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    rng = np.random.default_rng(seed)
    fixed = spnd.gaussian_filter(
        rng.standard_normal(shape).astype(np.float32), 2.0)
    fixed = (fixed - fixed.mean()) / (fixed.std() + 1e-6)
    disp = [spnd.gaussian_filter(rng.standard_normal(shape), smooth)
            for _ in range(ndim)]
    disp = [d / (np.abs(d).max() + 1e-6) * max_disp for d in disp]
    coords = np.indices(shape, dtype=np.float32)
    warped = [coords[i] + disp[i] for i in range(ndim)]
    moving = spnd.map_coordinates(fixed, warped, order=1, mode='nearest')
    return moving.astype(np.float32), fixed.astype(np.float32)


def aniso_pair(shape: Sequence[int], spacing: Sequence[float], seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    '''A same-grid non-rigid ``(moving, fixed, spacing)`` triple for the
    **anisotropic** demons / SyN points: the pair is the ``syn_pair`` smooth
    warp, registered on a grid whose voxels are anisotropic (e.g. ``1x1x3``).
    The op corrects the bias (a voxel-isotropic Gaussian / force is physically
    anisotropic) via its ``spacing`` argument; the refs get the matching
    ``diag([*spacing, 1])`` affine.'''
    moving, fixed = syn_pair(shape, seed)
    return moving, fixed, tuple(float(s) for s in spacing)


def motion_series(shape: Sequence[int], n_frames: int, seed: int = 0, *,
                  max_shift: float = 2.0) -> np.ndarray:
    '''A ``(T, *spatial)`` motion-corrupted series: one structured base volume
    under ``T`` small **known rigid** perturbations (scipy) + per-frame jitter
    -- a synthetic fMRI run for ``volreg`` to realign to a common reference.
    Each frame is the base rotated <=1.5 deg and shifted <= ``max_shift``
    voxels, so realignment must recover real (small) motion.'''
    import scipy.ndimage as spnd
    from scipy.spatial.transform import Rotation

    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    rng = np.random.default_rng(seed)
    base = spnd.gaussian_filter(
        rng.standard_normal(shape).astype(np.float32), 2.0)
    base = (base - base.mean()) / (base.std() + 1e-6)
    center = (np.asarray(shape) - 1) / 2.0
    frames = []
    for _ in range(int(n_frames)):
        rot = Rotation.from_euler(
            'xyz', rng.uniform(-1.5, 1.5, 3), degrees=True
        ).as_matrix()[:ndim, :ndim]
        shift = rng.uniform(-max_shift, max_shift, ndim)
        fr = spnd.affine_transform(
            base, rot, offset=center - rot @ center + shift,
            order=1, mode='nearest')
        fr = fr + 0.02 * rng.standard_normal(shape)  # acquisition jitter
        frames.append(fr.astype(np.float32))
    return np.stack(frames)


def bbr_boundary(shape: Sequence[int], n_points: int, seed: int = 0, *,
                 radius_frac: float = 0.3
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''A ``(moving, points, normals)`` triple for boundary-based registration:
    a bright smoothed sphere (interior bright, exterior dark) on a ``shape``
    grid, ``n_points`` samples on its surface with **outward** radial unit
    normals (so ``-normal`` points into the bright interior -- the Greve-Fischl
    convention ``bbr_cost`` expects: ``inside = q - step*n``).  The points are
    displaced by a small **planted** rigid offset so the boundary sits *off*
    the true edge at identity; a correct ``bbr_register`` seats it back (the
    recovery pin is ``cost_history[-1] < cost_history[0]``).'''
    import scipy.ndimage as spnd
    from scipy.spatial.transform import Rotation

    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    rng = np.random.default_rng(seed)
    center = (np.asarray(shape) - 1) / 2.0
    radius = radius_frac * float(min(shape))
    coords = np.indices(shape, dtype=np.float32)
    dist = np.sqrt(sum((coords[i] - center[i]) ** 2 for i in range(ndim)))
    moving = spnd.gaussian_filter(
        (dist < radius).astype(np.float32), 1.5).astype(np.float32)

    dirs = rng.standard_normal((int(n_points), ndim))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    pts_true = center + radius * dirs  # on the true boundary
    # planted small rigid offset (within BBR's narrow capture range).
    rot = Rotation.from_euler(
        'xyz', [2.0, 1.5, 1.0], degrees=True).as_matrix()[:ndim, :ndim]
    trans = np.array([1.0, -0.8, 0.6])[:ndim]
    points = (rot @ (pts_true - center).T).T + center + trans
    normals = (rot @ dirs.T).T
    return (moving, points.astype(np.float32), normals.astype(np.float32))


def ants_motion_correction(type_of_transform: str = 'Rigid'
                           ) -> Callable[..., Any]:
    '''ANTsPy ``motion_correction`` -- the volreg gold standard (the
    ``3dvolreg`` / ``mcflirt`` task, ITK-backed).  Realigns a ``(T, *spatial)``
    series to the per-volume mean (matching nitrix ``reference='mean'``).  ANTs
    realigns **frame-by-frame on CPU** (T sequential registrations), so its
    wall-clock is ~ ``T x per-frame`` -- the foil for nitrix's vmap-batched
    GPU realignment (the gap GROWS with T).  ``type_of_transform='Rigid'`` (not
    the ``'BOLDRigid'`` default, which adds BOLD-specific pre-steps nitrix does
    not).  Time is moved to the last axis (ANTs' x,y,z,t convention) and back.
    ants lazy / not jit-compiled (full wall-clock); returns the realigned
    series ``(T, *spatial)``.'''

    def run(series: Any) -> Any:
        import ants

        arr = np.asarray(series, np.float32)
        img = ants.from_numpy(np.moveaxis(arr, 0, -1))  # (T,*sp) -> (*sp,T)
        fixed = ants.from_numpy(arr.mean(axis=0))       # per-volume mean
        mc = ants.motion_correction(image=img, fixed=fixed,
                                    type_of_transform=type_of_transform)
        out = mc['motion_corrected'].numpy()            # (*sp, T)
        return np.moveaxis(out, -1, 0)                  # -> (T, *sp)

    return run


# -- AFNI / FSL command-line tools (the COMMUNITY realignment standards) ------
# Binaries, not Python pkgs: framework 'numpy' (base env has nibabel + nitrix),
# located via NPERF_AFNI_DIR / NPERF_FSL_DIR (absolute, not $PATH). NIfTI
# round-trip: write the (T,*spatial) series time-last (x,y,z,t), run, read.
# Spin-up: tools/setup_neuro_refs.sh (/scratch is ephemeral -> that's the
# recipe; see README "Community neuro reference tools").


def _nifti_roundtrip_moco(cmd_fn: Callable[[str, str, str], list],
                          series: Any, *, env: Any = None) -> Any:
    '''Shared NIfTI round-trip for a CLI motion-correction tool: write the
    ``(T, *spatial)`` series to a temp NIfTI (time last), build + run the
    tool's argv via ``cmd_fn(in_path, out_path, tmpdir)``, read the realigned
    series back as ``(T, *spatial)``.  Tools that realign to the mean reference
    match nitrix ``reference='mean'``.'''
    import os
    import subprocess
    import tempfile

    import nibabel as nib

    arr = np.asarray(series, np.float32)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        inp = os.path.join(d, 'in.nii.gz')
        out = os.path.join(d, 'moco.nii.gz')
        nib.save(nib.Nifti1Image(np.moveaxis(arr, 0, -1), np.eye(4)), inp)
        subprocess.run(cmd_fn(inp, out, d), check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        moco = np.asarray(nib.load(out).get_fdata(), np.float32)  # (*sp, T)
    return np.moveaxis(moco, -1, 0)                               # -> (T, *sp)


def afni_volreg() -> Callable[..., Any]:
    '''AFNI ``3dvolreg`` -- *the* canonical motion-realignment tool and a
    community standard (cf. the ``ants_motion_correction`` caveat that ANTs is
    seldom used for moco; 3dvolreg is hand-optimised C, much faster).  Realigns
    to the per-volume **mean** (an external base dataset, matching nitrix
    ``reference='mean'``).  Binary at ``$NPERF_AFNI_DIR`` (default
    ``/scratch/nperf/abin``); not jit (full wall-clock).'''
    import os

    abin = os.environ.get('NPERF_AFNI_DIR', '/scratch/nperf/abin')

    def run(series: Any) -> Any:
        import nibabel as nib

        def cmd(inp: str, out: str, d: str) -> list:
            base = os.path.join(d, 'base.nii.gz')
            mean = np.asarray(series, np.float32).mean(axis=0)
            nib.save(nib.Nifti1Image(mean, np.eye(4)), base)
            return [os.path.join(abin, '3dvolreg'), '-base', base,
                    '-prefix', out, '-overwrite', '-quiet', inp]

        return _nifti_roundtrip_moco(cmd, series)

    return run


def fsl_mcflirt() -> Callable[..., Any]:
    '''FSL ``mcflirt`` -- the other community motion-realignment standard (the
    FSL counterpart of AFNI 3dvolreg).  ``-meanvol`` realigns to the mean
    volume (matching nitrix ``reference='mean'``).  Binary at
    ``$NPERF_FSL_DIR/bin`` (default ``/scratch/nperf/fsl``); not jit.'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(series: Any) -> Any:
        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}

        def cmd(inp: str, out: str, d: str) -> list:
            # mcflirt appends .nii.gz to -out; point it at moco (sans suffix).
            return [os.path.join(fsldir, 'bin', 'mcflirt'), '-in', inp,
                    '-out', out[:-len('.nii.gz')], '-meanvol']

        return _nifti_roundtrip_moco(cmd, series, env=env)

    return run


# -- I/O floor no-ops: subtract the NIfTI round-trip from the walltime --------
# A NO-OP with the SAME write + subprocess + read as the real tool, but trivial
# compute -> its wall-clock IS the I/O floor (a harness artifact nitrix doesn't
# pay; the in-memory array is serialised to NIfTI only to feed the CLI tool).
# economic_report subtracts it: compute = tool_walltime - iofloor_walltime.


def afni_iofloor() -> Callable[..., Any]:
    '''AFNI ``3dcalc -expr a`` -- the identity (read 4D + write 4D, no
    registration), the I/O floor for ``afni_volreg`` (same round-trip).'''
    import os

    abin = os.environ.get('NPERF_AFNI_DIR', '/scratch/nperf/abin')

    def run(series: Any) -> Any:
        def cmd(inp: str, out: str, d: str) -> list:
            return [os.path.join(abin, '3dcalc'), '-a', inp, '-expr', 'a',
                    '-prefix', out, '-overwrite']

        return _nifti_roundtrip_moco(cmd, series)

    return run


def fsl_iofloor() -> Callable[..., Any]:
    '''FSL ``fslmaths -mul 1`` -- the identity (read 4D + write 4D), the I/O
    floor for ``fsl_mcflirt`` (same round-trip).'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(series: Any) -> Any:
        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}

        def cmd(inp: str, out: str, d: str) -> list:
            return [os.path.join(fsldir, 'bin', 'fslmaths'), inp, '-mul', '1',
                    out]

        return _nifti_roundtrip_moco(cmd, series, env=env)

    return run
