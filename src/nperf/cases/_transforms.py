# -*- coding: utf-8 -*-
"""Shared helpers for the Lie transform-exp family (batched, 3-D).

``nitrix.geometry.{rigid_exp, affine_exp, rigid_log}`` map between Lie
parameters and homogeneous ``(..., 4, 4)`` matrices, batched over leading dims
-- so the bench axis is the **batch size B** (a cohort of transforms / a
per-voxel local-affine field), and the matrices are tiny (4x4): an
embarrassingly-parallel throughput op.

The numpy reimpls are the fp64 oracle + CPU floor; the cupy ones (same
body, generic over ``xp``) are the on-target GPU bar.  Exact-convention
(verified vs nitrix: rigid_exp ~5e-7, affine_exp ~5e-5 = fp32 matrix_exp
round-off, rigid_log ~1e-7):

- ``rigid_exp(params)`` -- ``params[..., :3] = ω`` (axis-angle) -> Rodrigues
  SO(3) rotation; ``params[..., 3:6] = t`` placed directly.
- ``affine_exp(params)`` -- ``params[..., :9] = A`` (row-major gl(3) generator)
  -> ``expm(A)`` linear block; ``params[..., 9:12] = t`` direct.
- ``rigid_log(matrix)`` -- the inverse of ``rigid_exp``: principal SO(3) log of
  the rotation block + the translation column.  The inputs are generated from
  bounded ``ω`` (|ω| < ~1.5 rad) so they stay away from the ``θ = π`` log
  singularity (axis ill-defined there) -- the case exercises the well-posed
  interior, the regime users hit.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


# ---- input generators ----------------------------------------------------- #
def rigid_params(b: int, seed: int = 0) -> np.ndarray:
    '''``(B, 6)`` rigid Lie params: small rotation + larger translation.'''
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((b, 3)).astype(np.float32) * 0.5
    trans = rng.standard_normal((b, 3)).astype(np.float32) * 5.0
    return np.concatenate([omega, trans], axis=-1)


def affine_params(b: int, seed: int = 0) -> np.ndarray:
    '''``(B, 12)`` affine Lie params: small gl(3) generator + translation.'''
    rng = np.random.default_rng(seed)
    gen = rng.standard_normal((b, 9)).astype(np.float32) * 0.1
    trans = rng.standard_normal((b, 3)).astype(np.float32) * 5.0
    return np.concatenate([gen, trans], axis=-1)


def rigid_matrices(b: int, seed: int = 0) -> np.ndarray:
    '''``(B, 4, 4)`` valid rigid homogeneous matrices (= rigid_exp of params),
    the input for ``rigid_log``.'''
    return _rigid_exp(rigid_params(b, seed), np)


# ---- generic (numpy / cupy) reimplementations ----------------------------- #
def _skew3(w: Any, xp: Any) -> Any:
    z = xp.zeros(w.shape[:-1], dtype=w.dtype)
    ox, oy, oz = w[..., 0], w[..., 1], w[..., 2]
    return xp.stack([xp.stack([z, -oz, oy], -1),
                     xp.stack([oz, z, -ox], -1),
                     xp.stack([-oy, ox, z], -1)], -2)


def _rodrigues(w: Any, xp: Any) -> Any:
    '''Batched SO(3) exponential of an axis-angle vector (Rodrigues).'''
    th = xp.linalg.norm(w, axis=-1)[..., None, None]
    k = _skew3(w, xp)
    small = th < 1e-8
    a = xp.where(small, 1.0, xp.sin(th) / xp.where(small, 1.0, th))
    b = xp.where(small, 0.5, (1.0 - xp.cos(th)) / xp.where(small, 1.0, th**2))
    eye = xp.eye(3, dtype=w.dtype)
    return eye + a * k + b * xp.matmul(k, k)


def _expm_taylor(a: Any, xp: Any, terms: int = 20) -> Any:
    '''Batched matrix exponential via Taylor (exact for the bounded-norm gl(3)
    generators here -- an independent oracle for nitrix's scaling-squaring).'''
    n = a.shape[-1]
    eye = xp.broadcast_to(xp.eye(n, dtype=a.dtype), a.shape).copy()
    term = eye.copy()
    acc = eye.copy()
    for k in range(1, terms):
        term = xp.matmul(term, a) / k
        acc = acc + term
    return acc


def _so3_log(r: Any, xp: Any) -> Any:
    '''Principal axis-angle of a rotation (valid away from θ = π).'''
    cos = xp.clip((xp.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    th = xp.arccos(cos)
    w = xp.stack([r[..., 2, 1] - r[..., 1, 2],
                  r[..., 0, 2] - r[..., 2, 0],
                  r[..., 1, 0] - r[..., 0, 1]], -1)
    s = xp.sin(th)[..., None]
    small = xp.abs(s) < 1e-7
    return xp.where(small, 0.5 * w,
                    (th[..., None] / (2.0 * xp.where(small, 1.0, s))) * w)


def _homog(r: Any, t: Any, xp: Any) -> Any:
    b = r.shape[:-2]
    d = r.shape[-1]
    m = xp.zeros(b + (d + 1, d + 1), dtype=r.dtype)
    m[..., :d, :d] = r
    m[..., :d, d] = t
    m[..., d, d] = 1.0
    return m


def _rigid_exp(p: Any, xp: Any) -> Any:
    return _homog(_rodrigues(p[..., :3], xp), p[..., 3:6], xp)


def _affine_exp(p: Any, xp: Any) -> Any:
    a = p[..., :9].reshape(p.shape[:-1] + (3, 3))
    return _homog(_expm_taylor(a, xp), p[..., 9:12], xp)


def _rigid_log(m: Any, xp: Any) -> Any:
    return xp.concatenate([_so3_log(m[..., :3, :3], xp), m[..., :3, 3]], -1)


# ---- numpy fp64 oracles / cupy GPU refs ----------------------------------- #
def np_rigid_exp(p: Any) -> np.ndarray:
    return _rigid_exp(np.asarray(p), np)


def np_affine_exp(p: Any) -> np.ndarray:
    return _affine_exp(np.asarray(p), np)


def np_rigid_log(m: Any) -> np.ndarray:
    return _rigid_log(np.asarray(m), np)


def _cupy_ref(fn: Callable[[Any, Any], Any]) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        return fn(x, cp)

    return run


def cupy_rigid_exp() -> Callable[[Any], Any]:
    return _cupy_ref(_rigid_exp)


def cupy_affine_exp() -> Callable[[Any], Any]:
    return _cupy_ref(_affine_exp)


def cupy_rigid_log() -> Callable[[Any], Any]:
    return _cupy_ref(_rigid_log)
