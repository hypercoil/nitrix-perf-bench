# -*- coding: utf-8 -*-
"""Pure-numpy shape / sanity checks for the registration input generators
(`_register.py` additions) -- no nitrix, no jax, fast.  These pin only that the
planted inputs are well-formed (shapes, dtypes, a recoverable signal); the
end-to-end recovery is pinned in ``test_register_cases.py``.
"""
import numpy as np

from nperf.cases._register import (
    aniso_pair,
    bbr_boundary,
    syn_pair,
    warp_pair_cross_grid,
)


def test_cross_grid_shapes_affines():
    moving, fixed, a_m, a_f = warp_pair_cross_grid(
        (48, 48, 48), (56, 48, 40),
        fixed_spacing=(1.0, 1.0, 1.0), moving_spacing=(1.2, 1.0, 0.9))
    assert fixed.shape == (48, 48, 48) and moving.shape == (56, 48, 40)
    assert moving.dtype == np.float32 and fixed.dtype == np.float32
    assert a_m.shape == (4, 4) and a_f.shape == (4, 4)
    # the affines carry the anisotropic spacing on the diagonal.
    assert np.allclose(np.diag(a_m), [1.2, 1.0, 0.9, 1.0])
    # moving is a warped/resampled fixed -> it is non-trivial (not all zero).
    assert np.isfinite(moving).all() and moving.std() > 1e-3


def test_syn_pair_is_a_real_deformation():
    moving, fixed = syn_pair((32, 32, 32), seed=1)
    assert moving.shape == fixed.shape == (32, 32, 32)
    assert moving.dtype == np.float32
    # a genuine (small) deformation: differs from fixed but stays correlated.
    assert not np.allclose(moving, fixed)
    c = np.corrcoef(moving.ravel(), fixed.ravel())[0, 1]
    assert 0.3 < c < 0.999


def test_aniso_pair_returns_spacing():
    moving, fixed, spacing = aniso_pair((24, 24, 24), (1.0, 1.0, 3.0))
    assert moving.shape == fixed.shape == (24, 24, 24)
    assert spacing == (1.0, 1.0, 3.0)


def test_bbr_boundary_bright_sphere_outward_normals():
    moving, points, normals = bbr_boundary((40, 40, 40), 500, seed=0)
    assert points.shape == (500, 3) and normals.shape == (500, 3)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
    # interior (centre) is brighter than the corner (exterior).
    c = (np.asarray(moving.shape) - 1) // 2
    assert moving[c[0], c[1], c[2]] > moving[0, 0, 0] + 0.3
    # the boundary points sit at ~one radius from the centre (a sphere).
    center = (np.asarray(moving.shape) - 1) / 2.0
    r = np.linalg.norm(points - center, axis=1)
    assert r.std() / r.mean() < 0.2  # roughly spherical (planted offset aside)
