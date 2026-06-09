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
test.  ANTs is the task-level domain reference (``ants.registration``); it is
not jit-compiled (compiled C++), so the honest framing is nitrix
*steady* (post-compile) + its one-time compile, vs ANTs wall-clock.
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


def ants_rigid() -> Callable[..., Any]:
    '''ANTsPy rigid registration (the task-level domain reference); returns the
    warped moving.  ants lazy (only its refs env imports it); not jit-compiled,
    so its wall-clock is the full registration (no separate compile).'''

    def run(moving: Any, fixed: Any) -> Any:
        import ants

        f = ants.from_numpy(np.asarray(fixed, np.float32))
        m = ants.from_numpy(np.asarray(moving, np.float32))
        reg = ants.registration(fixed=f, moving=m, type_of_transform='Rigid')
        return reg['warpedmovout'].numpy()

    return run
