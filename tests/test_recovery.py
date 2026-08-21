# -*- coding: utf-8 -*-
"""Recovery scorer math (REGISTRATION_RECOVERY) -- validated synthetically,
independent of real data / nitrix: a known planted field, scored against
itself (perfect) and against a wrong field, must behave as the metric promises.
"""
import numpy as np

from nperf.cases._recovery import (
    RecoveryGT,
    identity_grid,
    jacobian_min,
    warp_by_field,
)


def _phantom(shape=(32, 32, 32)):
    g = identity_grid(shape)
    c = (np.asarray(shape) - 1) / 2
    r = np.linalg.norm(g - c, axis=-1)
    # a smooth textured blob (texture so NCC/TRE are sensitive, not flat)
    fixed = (np.exp(-(r ** 2) / (2 * (min(shape) * 0.22) ** 2))
             * (1 + 0.3 * np.cos(g[..., 0] * 0.7))).astype(np.float64)
    mask = r < min(shape) * 0.42
    return fixed, mask


def _planted(gt_disp):
    fixed, mask = _phantom()
    gt = np.zeros(fixed.shape + (3,))
    for i, d in enumerate(gt_disp):
        gt[..., i] = d
    moving = warp_by_field(fixed, -gt)          # so warp(moving, gt) == fixed
    return RecoveryGT(fixed=fixed, moving=moving, gt_field=gt, mask=mask,
                      spacing=2.0)


def test_perfect_recovery_scores_near_zero():
    gt = _planted([1.5, -0.8, 0.4])
    s = gt.score(gt.gt_field)                    # recovered == truth
    assert s['recovery_tre'] < 1e-6, s
    assert s['recovery_warp'] < 1e-6, s
    # ~1.0 up to the unavoidable double-interpolation blur (moving is built by
    # warping fixed, then warped back) -- not a recovery error.
    assert s['recovery_ncc'] > 0.99, s
    assert s['recovery_jacmin'] > 0.5, s         # identity-ish: detJ ~ 1


def test_did_nothing_scores_the_planted_magnitude():
    gt = _planted([1.5, 0.0, 0.0])
    s = gt.score(np.zeros_like(gt.gt_field))     # recovered nothing
    # TRE == ||planted disp|| (1.5 vox) * spacing (2 mm) = 3 mm
    assert abs(s['recovery_tre'] - 3.0) < 1e-6, s
    # and it aligns worse than the true recovery
    assert s['recovery_ncc'] < gt.score(gt.gt_field)['recovery_ncc']


def test_jacobian_flags_folding():
    shape = (24, 24, 24)
    g = identity_grid(shape)
    fold = np.zeros(shape + (3,))
    fold[..., 0] = -2.0 * g[..., 0]              # x -> -x : detJ < 0
    assert jacobian_min(fold) < 0
