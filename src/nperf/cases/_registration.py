# -*- coding: utf-8 -*-
"""Shared helpers for the registration / deformation-field family.

``nitrix.geometry.{jacobian_displacement, jacobian_det_displacement,
integrate_velocity_field}`` operate on a channel-last vector field
``(*spatial, ndim)`` with ``len(spatial) == ndim``:

- **jacobian_displacement**: per-point Jacobian ``J = I + ∇u`` of the
  deformation ``φ = id + u``, via central differences with the ``'nearest'``
  boundary (edge-clamped, denominator ``2·spacing`` *even at the one-sided
  boundary cell* -- the voxelmorph convention). This matches ``numpy.gradient``
  in the interior but NOT at the boundary (numpy.gradient uses a first-order
  one-sided diff with denominator ``1·spacing`` there), so the right oracle is
  a reimplementation of nitrix's exact roll-based central diff, verified equal
  to 0.0 in fp64 -- not ``numpy.gradient``.
- **jacobian_det_displacement**: ``det(I + ∇u)`` per point (closed-form Sarrus
  for d=3); the folding-detection QA scalar.
- **integrate_velocity_field**: the diffeomorphic exponential map by
  scaling-and-squaring (voxelmorph ``VecInt``): ``φ = v/2ⁿ`` then
  ``φ ← φ + φ∘(id+φ)`` for ``n_steps`` doublings, the composition being a
  linear-interpolation warp. The reference reuses **scipy.ndimage.map_
  coordinates** (order=1, mode='nearest') -- the genuine domain-tool core --
  for the composition; verified equal to ~5e-16 in fp64.

The numpy refs are the fp64 oracle + CPU floor; the cupy refs reimplement the
same roll-diff / use ``cupyx.scipy.ndimage.map_coordinates`` (a fair
kernel-vs-kernel GPU bar). All GPU-pure (no solver). scipy.ndimage is a core
dep; cupy / cupyx are lazy (their worker only).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np


def displacement_input(
    spatial: Sequence[int], ndim: int = 3, seed: int = 0,
    scale: float = 0.1,
) -> np.ndarray:
    '''A small random displacement field ``(*spatial, ndim)`` (small so the
    Jacobian determinant stays positive -- a realistic non-folding warp).'''
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(tuple(spatial) + (ndim,)) * scale).astype(
        np.float32)


def _central_diff(field: Any, ax: int, xp: Any, spacing: float = 1.0) -> Any:
    '''Roll-based central difference with edge-clamped (``'nearest'``)
    boundary, denominator ``2·spacing`` everywhere (nitrix's convention).'''
    nxt = xp.roll(field, -1, ax)
    prv = xp.roll(field, +1, ax)
    n = field.ndim
    L = field.shape[ax]
    sl_last = [slice(None)] * n
    sl_last[ax] = L - 1
    sl_first = [slice(None)] * n
    sl_first[ax] = 0
    nxt[tuple(sl_last)] = field[tuple(sl_last)]
    prv[tuple(sl_first)] = field[tuple(sl_first)]
    return (nxt - prv) / (2.0 * spacing)


def _jacobian(u: Any, xp: Any) -> Any:
    '''``J[..., i, j] = δ_ij + ∂u_i/∂x_j`` via central differences.'''
    d = u.shape[-1]
    cols = [_central_diff(u, -(d + 1) + j, xp) for j in range(d)]
    return xp.stack(cols, -1) + xp.eye(d, dtype=u.dtype)


def _jac_det(u: Any, xp: Any) -> Any:
    '''``det(I + ∇u)`` per point; closed-form Sarrus for d=3 (the case the
    bench exercises).'''
    j = _jacobian(u, xp)
    a = j[..., 0, 0]
    b = j[..., 0, 1]
    c = j[..., 0, 2]
    d_ = j[..., 1, 0]
    e = j[..., 1, 1]
    f = j[..., 1, 2]
    g = j[..., 2, 0]
    h = j[..., 2, 1]
    i = j[..., 2, 2]
    return a * (e * i - f * h) - b * (d_ * i - f * g) + c * (d_ * h - e * g)


def _sample(field: Any, coords: Any, xp: Any, ndimage: Any,
            mode: str = 'nearest') -> Any:
    '''Linear-interpolation warp: sample channel-last ``field`` at absolute
    ``coords`` (``(*spatial, ndim)``), one channel at a time.'''
    ndim = coords.shape[-1]
    ct = xp.moveaxis(coords, -1, 0).reshape(ndim, -1)
    sp = field.shape[:-1]
    out = [ndimage.map_coordinates(field[..., c], ct, order=1,
                                   mode=mode).reshape(sp)
           for c in range(field.shape[-1])]
    return xp.stack(out, -1)


def _integrate(v: Any, xp: Any, ndimage: Any, n_steps: int = 7,
               mode: str = 'nearest') -> Any:
    '''Scaling-and-squaring exponential map (voxelmorph VecInt).'''
    sp = v.shape[:-1]
    idg = xp.stack(
        xp.meshgrid(*[xp.arange(s, dtype=v.dtype) for s in sp],
                    indexing='ij'),
        -1)
    phi = v / float(2 ** n_steps)
    for _ in range(n_steps):
        phi = phi + _sample(phi, idg + phi, xp, ndimage, mode)
    return phi


# ---- numpy floors / fp64 oracles -----------------------------------------


def np_jacobian(u: Any) -> np.ndarray:
    return _jacobian(np.asarray(u), np)


def np_jac_det(u: Any) -> np.ndarray:
    return _jac_det(np.asarray(u), np)


def np_integrate(v: Any) -> np.ndarray:
    import scipy.ndimage as ndi

    return _integrate(np.asarray(v), np, ndi)


# ---- cupy GPU references --------------------------------------------------


def cupy_jacobian() -> Callable[[Any], Any]:
    def run(u: Any) -> Any:
        import cupy as cp

        return _jacobian(u, cp)

    return run


def cupy_jac_det() -> Callable[[Any], Any]:
    def run(u: Any) -> Any:
        import cupy as cp

        return _jac_det(u, cp)

    return run


def cupy_integrate() -> Callable[[Any], Any]:
    def run(v: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        return _integrate(v, cp, cndi)

    return run


def jacobian_sizes(d: int) -> Tuple[int, int, int]:
    '''A (d, d, d) spatial cube for the 3-D registration cases.'''
    return (d, d, d)


# ---- explicit regularisers (nitrix.register penalties) -------------------- #
# Exact-convention reimplementations of the displacement-field penalties (all
# reuse the same roll-based central diff above, so they match nitrix's gradient
# / boundary convention, not numpy.gradient's). Generic over xp -> one body
# serves the numpy fp64 oracle + the cupy GPU ref. Default reduction 'mean'
# (the training-loss form users add to a loss).


def _mean(x: Any, xp: Any) -> Any:
    return xp.mean(x)


def _gradient_smoothness(u: Any, xp: Any) -> Any:
    '''``mean ‖∇u‖²`` -- squared Frobenius of the displacement Jacobian.'''
    d = u.shape[-1]
    grad_u = _jacobian(u, xp) - xp.eye(d, dtype=u.dtype)
    return _mean(xp.sum(grad_u ** 2, axis=(-2, -1)), xp)


def _bending_energy(u: Any, xp: Any) -> Any:
    '''``mean ‖∇²u‖²`` -- squared Frobenius of the per-voxel Hessian (a second
    central diff of each ``∇u`` component).'''
    d = u.shape[-1]
    grad_u = _jacobian(u, xp) - xp.eye(d, dtype=u.dtype)
    flat = grad_u.reshape(u.shape[:-1] + (d * d,))
    comps = [xp.stack([_central_diff(flat[..., c], ax, xp)
                       for ax in range(d)], -1)
             for c in range(d * d)]
    hess = xp.stack(comps, -2)  # (*spatial, d*d, d)
    return _mean(xp.sum(hess ** 2, axis=(-2, -1)), xp)


def _folding(u: Any, xp: Any) -> Any:
    '''``mean relu(-det J)`` of ``J = I + ∇u`` -- the folding penalty.'''
    return _mean(xp.maximum(-_jac_det(u, xp), 0.0), xp)


def np_gradient_smoothness(u: Any) -> np.ndarray:
    return np.asarray(_gradient_smoothness(np.asarray(u), np))


def np_bending_energy(u: Any) -> np.ndarray:
    return np.asarray(_bending_energy(np.asarray(u), np))


def np_folding(u: Any) -> np.ndarray:
    return np.asarray(_folding(np.asarray(u), np))


def _cupy_penalty(fn: Callable[[Any, Any], Any]) -> Callable[[Any], Any]:
    def run(u: Any) -> Any:
        import cupy as cp

        return fn(u, cp)

    return run


def cupy_gradient_smoothness() -> Callable[[Any], Any]:
    return _cupy_penalty(_gradient_smoothness)


def cupy_bending_energy() -> Callable[[Any], Any]:
    return _cupy_penalty(_bending_energy)


def cupy_folding() -> Callable[[Any], Any]:
    return _cupy_penalty(_folding)


# ---- spatial_gradient / compose_velocity / invert_displacement ------------ #
# More of the deformation-field algebra (deformation.py). Same roll-based
# central diff -> the numpy/cupy reimpls match nitrix's exact convention.


def scalar_field_input(spatial: Sequence[int], seed: int = 0) -> np.ndarray:
    '''A scalar field ``(*spatial,)`` -- the image for ``spatial_gradient``.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal(tuple(spatial)).astype(np.float32)


def _spatial_grad(field: Any, xp: Any) -> Any:
    '''``∂field/∂x_j`` per spatial axis -> ``(*spatial, ndim)`` (central).'''
    ndim = field.ndim
    return xp.stack([_central_diff(field, ax, xp) for ax in range(ndim)], -1)


def _compose_velocity2(v: Any, u: Any, xp: Any) -> Any:
    '''BCH order-2 ``v + u + ½[v,u]``; ``[v,u] = (v·∇)u - (u·∇)v``.'''
    d = u.shape[-1]
    du = _jacobian(u, xp) - xp.eye(d, dtype=u.dtype)
    dv = _jacobian(v, xp) - xp.eye(d, dtype=v.dtype)
    du_v = xp.einsum('...ij,...j->...i', du, v)
    dv_u = xp.einsum('...ij,...j->...i', dv, u)
    return v + u + 0.5 * (du_v - dv_u)


def _invert_disp(s: Any, xp: Any, ndimage: Any, tol: float = 1e-5,
                 max_iter: int = 50) -> Any:
    '''Inverse displacement by Picard fixed point ``s_inv = -s∘(id+s_inv)``,
    iterated to a relative ``tol`` (nitrix's ``fixed_point_solve`` convention;
    the per-iteration host check syncs cupy -- the naive iterative reimpl that
    nitrix's jitted ``lax.while_loop`` improves on).'''
    sp = s.shape[:-1]
    idg = xp.stack(
        xp.meshgrid(*[xp.arange(n, dtype=s.dtype) for n in sp], indexing='ij'),
        -1)
    s_inv = xp.zeros_like(s)
    for _ in range(max_iter):
        new = -_sample(s, idg + s_inv, xp, ndimage)
        moved = float(xp.sqrt(xp.sum((new - s_inv) ** 2)))
        scale = float(xp.sqrt(xp.sum(new ** 2)))
        s_inv = new
        if moved <= tol * (scale + 1e-12):
            break
    return s_inv


def np_spatial_gradient(field: Any) -> np.ndarray:
    return np.asarray(_spatial_grad(np.asarray(field), np))


def np_compose_velocity(v: Any, u: Any) -> np.ndarray:
    return np.asarray(_compose_velocity2(np.asarray(v), np.asarray(u), np))


def np_invert_displacement(s: Any) -> np.ndarray:
    import scipy.ndimage as ndi

    return np.asarray(_invert_disp(np.asarray(s), np, ndi))


def cupy_spatial_gradient() -> Callable[[Any], Any]:
    def run(field: Any) -> Any:
        import cupy as cp

        return _spatial_grad(field, cp)

    return run


def cupy_compose_velocity() -> Callable[[Any, Any], Any]:
    def run(v: Any, u: Any) -> Any:
        import cupy as cp

        return _compose_velocity2(v, u, cp)

    return run


def cupy_invert_displacement() -> Callable[[Any], Any]:
    def run(s: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi

        return _invert_disp(s, cp, cndi)

    return run
