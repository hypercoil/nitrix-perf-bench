# -*- coding: utf-8 -*-
"""Real anatomical images for the registration cases (vs synthetic noise).

Multiple requests for benchmarks on **real anatomy** -- real edges, real
intensity statistics, the actual difficulty registration faces -- not the
smoothed-noise ``warp_pair``.  Source: **nilearn** datasets (the bundled MNI152
T1 template needs *no download*; ``fetch_oasis_vbm`` adds real *individuals*).

The success-metric challenge is resolved by **planting a known warp on the real
image** (Tier 1): the anatomy is real (realistic difficulty) but the
ground-truth transform is exact (we applied it with scipy), so the existing
recovery pins (ncc improvement, param/TRE recovery) carry over.  Inter-subject
(no GT warp -> Dice / MI) is a later realism tier.

**Env / reproducibility.**  The real array is cached as a plain ``.npy`` under
``$NPERF_REAL_DATA`` (default ``/scratch/nperf/real_anatomy``): the first call
materialises it via nilearn, every later call (including the **ANTs / dipy ref
envs, which have no nilearn**) reads the numpy cache.  ``/scratch`` is
ephemeral, so the cache regenerates from nilearn on demand -- pre-warm it once
(``load_mni152()`` in the base env) before a ref-env sweep.  ``nilearn`` itself
is only imported on a cache miss.
"""
from __future__ import annotations

import os
from typing import Sequence, Tuple

import numpy as np

_CACHE = os.environ.get('NPERF_REAL_DATA', '/scratch/nperf/real_anatomy')


def load_mni152(resolution: int = 2) -> np.ndarray:
    '''The MNI152 T1 template -- a real (population-averaged) brain, normalised
    to zero-mean/unit-std fp32.  ``resolution`` in mm (2 -> ~99x117x95, 1 ->
    ~197x233x189).  Cached to ``.npy`` so the ref envs read numpy, not nilearn
    (which is imported only to fill the cache).'''
    path = os.path.join(_CACHE, f'mni152_{resolution}mm.npy')
    if os.path.exists(path):
        return np.load(path)
    from nilearn import datasets as ds  # cache-miss only

    arr = np.asarray(
        ds.load_mni152_template(resolution=resolution).get_fdata(), np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    os.makedirs(_CACHE, exist_ok=True)
    np.save(path, arr)
    return arr


# A small **background noise floor**: real scans have acquisition noise, so the
# MNI152 template's artificially-uniform (zero) background is itself
# unrealistic -- adding independent noise to each "acquisition" is *more*
# faithful, and it also breaks the uniform 0/0 that NaNs nitrix's demons ESM
# force on real images (filed: register-demons-force-divide-by-zero). Applied
# to moving AND fixed
# independently (two noisy acquisitions of the same anatomy).
_NOISE = 0.02


def _add_noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (img + _NOISE * rng.standard_normal(img.shape)).astype(np.float32)


def real_warp_pair(resolution: int = 2, seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    '''A ``(moving, fixed)`` pair on **real anatomy**: ``fixed`` is the MNI152
    T1, ``moving`` is it under a small known **rigid** warp (scipy, independent
    of nitrix) -- the real-anatomy analog of ``warp_pair``.  ~3 deg rotation +
    few-voxel shift (within capture range), each with an independent realistic
    background noise floor (see ``_NOISE``).'''
    import scipy.ndimage as spnd
    from scipy.spatial.transform import Rotation

    fixed = load_mni152(resolution)
    rng = np.random.default_rng(seed)
    rot = Rotation.from_euler('xyz', [3.0, 2.0, 1.5], degrees=True).as_matrix()
    center = (np.asarray(fixed.shape) - 1) / 2.0
    offset = center - rot @ center + rng.uniform(-2.0, 2.0, 3)
    moving = spnd.affine_transform(fixed, rot, offset=offset, order=1,
                                   mode='nearest')
    return _add_noise(moving, rng), _add_noise(fixed, rng)


def real_syn_pair(resolution: int = 2, seed: int = 0, *,
                  max_disp: float = 8.0, smooth: float = 5.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    '''A real-anatomy ``(moving, fixed)`` pair under a smooth **non-rigid**
    warp (the deformable-recovery analog of ``syn_pair``, for SyN / demons):
    ``fixed`` is the MNI152 T1, ``moving`` is it pushed by a low-frequency
    displacement field (gaussian-smoothed, scaled to ``max_disp`` voxels) via
    scipy ``map_coordinates`` -- independent of nitrix -- each with the noise
    floor.'''
    import scipy.ndimage as spnd

    fixed = load_mni152(resolution)
    shape = fixed.shape
    ndim = fixed.ndim
    rng = np.random.default_rng(seed)
    disp = [spnd.gaussian_filter(rng.standard_normal(shape), smooth)
            for _ in range(ndim)]
    disp = [d / (np.abs(d).max() + 1e-6) * max_disp for d in disp]
    coords = np.indices(shape, dtype=np.float32)
    warped = [coords[i] + disp[i] for i in range(ndim)]
    moving = spnd.map_coordinates(fixed, warped, order=1, mode='nearest')
    return _add_noise(moving, rng), _add_noise(fixed, rng)


def mni152_shape(resolution: int = 2) -> Sequence[int]:
    '''The MNI152 template's spatial shape at ``resolution`` mm (for sizing /
    labelling without loading the full array when the cache exists).'''
    return tuple(int(s) for s in load_mni152(resolution).shape)


# --- raw-intensity anatomy for the bias / intensity ops --------------------
# ``load_mni152`` z-scores (signed, ~N(0,1)); but N4 works in *log* space and
# intensity_normalize clips *raw* percentiles, so they need the positive
# acquisition intensity scale.  Cached separately (``..._raw.npy``).

def load_mni152_raw(resolution: int = 2) -> np.ndarray:
    '''The MNI152 T1 template at **raw** (positive) intensities -- the
    acquisition scale the bias / intensity ops operate on, unlike the z-scored
    ``load_mni152`` (signed).  Same ``.npy`` cache discipline (nilearn imported
    only on a miss).'''
    path = os.path.join(_CACHE, f'mni152_{resolution}mm_raw.npy')
    if os.path.exists(path):
        return np.load(path)
    from nilearn import datasets as ds  # cache-miss only

    arr = np.asarray(
        ds.load_mni152_template(resolution=resolution).get_fdata(), np.float32)
    os.makedirs(_CACHE, exist_ok=True)
    np.save(path, arr)
    return arr


def _smooth_bias(shape: Sequence[int], rng: np.random.Generator,
                 smooth: float = 6.0, lo: float = 0.6, hi: float = 1.7
                 ) -> np.ndarray:
    '''A smooth low-frequency multiplicative bias field in ``[lo, hi]`` (a
    gaussian-smoothed random field) -- the planted intensity inhomogeneity that
    is N4's *known recoverable truth*.'''
    import scipy.ndimage as spnd

    f = spnd.gaussian_filter(rng.standard_normal(tuple(shape)), smooth)
    f = (f - f.min()) / (f.max() - f.min() + 1e-6)  # -> [0, 1]
    return (lo + (hi - lo) * f).astype(np.float32)


def real_bias_phantom(resolution: int = 2, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    '''Real MNI152 T1 (raw positive intensities) under a **known** smooth
    multiplicative bias field -- the real-anatomy analog of the synthetic N4
    ``phantom`` (``cases/_itk.py``).  The anatomy/edges are real (the INU
    difficulty N4 actually faces) but the bias is *planted* (recoverable truth
    -> ``real_planted``).  Returns ``(observed, mask)`` with the same contract
    as ``phantom`` (mask = the brain region; obs is positive within it, so the
    N4 log-domain fit is well-posed).'''
    brain = load_mni152_raw(resolution)
    rng = np.random.default_rng(seed)
    mask = (brain > 0.05 * float(brain.max())).astype(np.float32)
    bias = _smooth_bias(brain.shape, rng)
    obs = brain * bias
    obs = obs + (rng.normal(0, 0.01 * float(brain.max()), brain.shape)
                 .astype(np.float32) * mask)
    return obs.astype(np.float32), mask


def real_brain_slice(resolution: int = 1) -> np.ndarray:
    '''A real 2-D brain slice in ``[0, 1]`` -- the central axial slice of the
    MNI152 T1, min-max rescaled (the ``[0, 1]`` contract ``bilateral_image``
    provides).  Real anatomy/edges = the *actual* edge-preserving-smoothing
    problem (``real_full``: no planted truth -- the filtered image is the
    deliverable, parity vs ITK is the bar).'''
    vol = load_mni152_raw(resolution)
    sl = np.asarray(vol[:, :, vol.shape[2] // 2], np.float32)
    lo, hi = float(sl.min()), float(sl.max())
    return ((sl - lo) / (hi - lo + 1e-6)).astype(np.float32)
