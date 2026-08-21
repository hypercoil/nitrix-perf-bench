# -*- coding: utf-8 -*-
"""Registration recovery-quality scoring (REGISTRATION_RECOVERY).

Planted-warp benchmarks score *how well a tool recovered a known transform*,
beside the speed.  The trick around the "no shared oracle" problem (nitrix /
ANTs / dipy converge to different transforms) is a **known ground truth**: a
transform of realistic magnitude is planted on a real image to make the moving
image, so the true fixed->moving displacement field is known exactly.

**Uniform scoring via the recovered field.** Each baseline's recovery output
is its **recovered forward displacement field on the fixed grid, in voxels**
(``(*spatial, ndim)``; ``field[x] = where fixed voxel x maps to in moving``).
nitrix returns the transform/field natively; the community wrappers extract
theirs (``_register``). Everything below derives from one field, so nitrix
and the refs are scored on identical axes:

- ``recovery_ncc``  -- warped-moving vs fixed (the uniform alignment column).
- ``recovery_tre``  -- median Target Registration Error (mm) at landmarks:
  ``||recovered_disp(landmark) - true_disp(landmark)||`` -- the field-standard
  geometric accuracy.
- ``recovery_warp`` -- RMS ``||field - gt_field||`` over the brain mask (mm).
- ``recovery_jacmin`` -- min Jacobian determinant of the mapping (``<0`` ==
  folding; a tool that "aligns" by folding is caught here, which TRE alone
  misses).

All host-numpy (scores the host ``out_host``); ``spacing`` converts voxels->mm.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:                                            # SciPy is a host-side dep
    from scipy.ndimage import map_coordinates
except Exception:                               # pragma: no cover
    map_coordinates = None


def identity_grid(shape) -> np.ndarray:
    '''The (*shape, ndim) array of integer voxel coordinates.'''
    idx = np.indices(tuple(int(s) for s in shape), dtype=np.float64)
    return np.moveaxis(idx, 0, -1)              # (*shape, ndim)


def matrix_to_field(matrix: np.ndarray, shape) -> np.ndarray:
    '''A homogeneous voxel->voxel affine ``matrix`` ((d+1,d+1)) as a forward
    displacement field on ``shape``: ``field[x] = matrix @ x - x``.'''
    m = np.asarray(matrix, np.float64)
    ndim = len(shape)
    grid = identity_grid(shape).reshape(-1, ndim)               # (N, d)
    homog = np.concatenate([grid, np.ones((grid.shape[0], 1))], 1)
    mapped = homog @ m[:ndim].T                                 # (N, d)
    return (mapped - grid).reshape(*shape, ndim)


def sample_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''Tri-linearly sample a ``(*spatial, ndim)`` displacement field at
    ``points`` ((K, ndim) voxel coords) -> (K, ndim) displacements.'''
    ndim = field.shape[-1]
    coords = points.T                                            # (ndim, K)
    return np.stack(
        [map_coordinates(field[..., c], coords, order=1, mode='nearest')
         for c in range(ndim)], axis=-1)


def warp_by_field(image: np.ndarray, field: np.ndarray) -> np.ndarray:
    '''Resample ``image`` by a forward displacement ``field`` (fixed grid):
    ``warped[x] = image[x + field[x]]``.'''
    grid = identity_grid(image.shape)
    src = grid + field                                          # where to read
    coords = np.moveaxis(src, -1, 0)                   # (ndim, *spatial)
    return map_coordinates(image, coords, order=1, mode='nearest')


def ncc(a: np.ndarray, b: np.ndarray,
        mask: Optional[np.ndarray] = None) -> float:
    '''Global Pearson correlation (optionally within ``mask``).'''
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel() > 0
        a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / den) if den > 0 else 0.0


def jacobian_min(
    field: np.ndarray, mask: Optional[np.ndarray] = None,
) -> float:
    '''Minimum determinant of the mapping Jacobian ``J = I + grad(field)``
    (``<0`` => folding).  3-D only; computed over ``mask`` if given.'''
    ndim = field.shape[-1]
    # grad[i][j] = d field_i / d x_j
    g = [[np.gradient(field[..., i], axis=j) for j in range(ndim)]
         for i in range(ndim)]
    jac = np.empty(field.shape[:-1] + (ndim, ndim), np.float64)
    for i in range(ndim):
        for j in range(ndim):
            jac[..., i, j] = g[i][j] + (1.0 if i == j else 0.0)
    det = np.linalg.det(jac)
    if mask is not None:
        det = det[np.asarray(mask) > 0]
    return float(det.min())


def landmark_grid(mask: np.ndarray, n_per_axis: int = 8) -> np.ndarray:
    '''A roughly-even grid of ``n_per_axis`` landmark voxel coords inside the
    brain ``mask`` (the foreground where TRE is meaningful).'''
    ndim = mask.ndim
    axes = [np.linspace(0.12, 0.88, n_per_axis) * (s - 1)
            for s in mask.shape]
    pts = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing='ij')], -1)
    # keep only landmarks that land in the mask (nearest-voxel test).
    idx = tuple(np.clip(np.round(pts[:, d]).astype(int), 0, mask.shape[d] - 1)
                for d in range(ndim))
    return pts[np.asarray(mask)[idx] > 0]


class RecoveryGT:
    '''Captured ground truth + scorer for one planted-warp recovery point.

    ``score(field)`` -> the recovery metric dict.  Built once in a case's
    ``_build``; the ``BuiltPoint.recovery`` closure dispatches each baseline's
    recovered field into it.'''

    def __init__(self, *, fixed: np.ndarray, moving: np.ndarray,
                 gt_field: np.ndarray, mask: np.ndarray,
                 spacing: float = 1.0, n_landmarks: int = 8) -> None:
        self.fixed = np.asarray(fixed, np.float64)
        self.moving = np.asarray(moving, np.float64)
        self.gt_field = np.asarray(gt_field, np.float64)
        self.mask = np.asarray(mask)
        self.spacing = float(spacing)
        self.landmarks = landmark_grid(self.mask, n_landmarks)
        self._gt_at_lm = sample_field(self.gt_field, self.landmarks)

    def score(self, field: Any) -> Dict[str, float]:
        field = np.asarray(field, np.float64)
        if field.shape != self.gt_field.shape:
            raise ValueError(
                f'recovered field {field.shape} != gt {self.gt_field.shape}')
        warped = warp_by_field(self.moving, field)
        d_rec = sample_field(field, self.landmarks)
        tre_vox = np.linalg.norm(d_rec - self._gt_at_lm, axis=1)
        warp_vox = np.sqrt(((field - self.gt_field) ** 2).sum(-1))[
            self.mask > 0]
        return {
            'recovery_ncc': ncc(warped, self.fixed, self.mask),
            'recovery_tre': float(np.median(tre_vox)) * self.spacing,
            'recovery_warp': float(np.sqrt((warp_vox ** 2).mean()))
            * self.spacing,
            'recovery_jacmin': jacobian_min(field, self.mask),
        }
