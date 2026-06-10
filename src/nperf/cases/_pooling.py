# -*- coding: utf-8 -*-
"""Shared helpers for the N-D max-pool / unpool pair (3-D, NCHW-style).

``nitrix.morphology.max_pool_with_indices_nd`` takes ``(*batch, C, *spatial)``
and returns ``(values, indices)`` -- the windowed max + the **global flat
spatial index** of each argmax; ``max_unpool_nd`` scatters pooled values back
to those indices in a zeroed grid.

The numpy reimplementations are the fp64 oracle + CPU floor; cupy is the GPU
bar.  Verified vs nitrix: pool values == numpy windowed-max (exact), unpool ==
numpy scatter (exact).  The references compute the **argmax too** (and the
nitrix baseline keeps the indices live via ``+ 0*idx``) so all three do the
same with-indices work -- a fair comparison, not max-only vs max+argmax.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

_B, _C, _P = 2, 3, 2  # batch, channels, pool size (non-overlapping 2^3 blocks)


def pool_input(d: int, seed: int = 0) -> np.ndarray:
    '''A ``(B, C, d, d, d)`` field to max-pool.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_B, _C, d, d, d)).astype(np.float32)


def _pool3d(x: Any, xp: Any, p: int = _P) -> Any:
    '''Windowed max over non-overlapping p^3 blocks, with the argmax computed +
    kept live (the with-indices op computes both); returns the max.'''
    b, c, d, h, w = x.shape
    xr = x.reshape(b, c, d // p, p, h // p, p, w // p, p)
    m = xr.max(axis=(3, 5, 7))
    a = (xr.transpose(0, 1, 2, 4, 6, 3, 5, 7)
           .reshape(b, c, d // p, h // p, w // p, p ** 3).argmax(-1))
    return m + (a * 0).astype(m.dtype)


def np_pool(x: Any) -> np.ndarray:
    return np.asarray(_pool3d(np.asarray(x), np))


def cupy_pool() -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        return _pool3d(x, cp)

    return run


def pool_for_unpool(d: int, seed: int = 0) -> tuple:
    '''``(values, indices)`` for the unpool input -- the real output of nitrix
    ``max_pool_with_indices_nd`` (valid indices in its flat convention).'''
    import jax.numpy as jnp
    from nitrix.morphology import max_pool_with_indices_nd

    x = pool_input(d, seed)
    m, i = max_pool_with_indices_nd(jnp.asarray(x), pool_size=_P,
                                    spatial_rank=3)
    return np.asarray(m), np.asarray(i)


def _unpool(values: Any, indices: Any, out_spatial: Sequence[int],
            xp: Any) -> Any:
    '''Scatter ``values`` to their flat ``indices`` in a zeroed grid.'''
    b, c = values.shape[:2]
    n = int(np.prod(out_spatial))
    out = xp.zeros((b, c, n), dtype=values.dtype)
    vflat = values.reshape(b, c, -1)
    iflat = indices.reshape(b, c, -1).astype(xp.int64)
    xp.put_along_axis(out, iflat, vflat, axis=2)
    return out.reshape((b, c) + tuple(out_spatial))


def np_unpool(out_spatial: Sequence[int]) -> Callable[[Any, Any], Any]:
    def run(values: Any, indices: Any) -> np.ndarray:
        return np.asarray(_unpool(np.asarray(values), np.asarray(indices),
                                  out_spatial, np))

    return run


def cupy_unpool(out_spatial: Sequence[int]) -> Callable[[Any, Any], Any]:
    def run(values: Any, indices: Any) -> Any:
        import cupy as cp

        return _unpool(values, indices, out_spatial, cp)

    return run
