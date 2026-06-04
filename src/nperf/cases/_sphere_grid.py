# -*- coding: utf-8 -*-
"""Shared helpers for the parameterised-sphere (equirectangular grid) family.

``nitrix.geometry.{sphere_grid_pad_2d, sphere_grid_unpad_2d}`` pad / unpad a 2D
equirectangular sphere image with its non-trivial topology: the longitudinal
(width) axis wraps circularly, while the latitudinal (height) axis is
pole-bounded -- the top/bottom pads come from the rows just inside each pole,
flipped vertically and rolled longitudinally by ``W/2`` ("over the pole" lands
on the opposite longitude half). The unpad is the inverse slice.

A nitrix-specific topology (no external library expresses the pole-flip as a
single boundary mode), so the references are a numpy reimplementation of the
exact pad/slice (verified equal to 0.0 in fp64, with an exact
``unpad(pad(x)) == x`` round-trip) + a CuPy GPU ref. Pure gather/slice/roll, so
GPU-pure.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def sphere_grid_input(h: int, seed: int = 0) -> np.ndarray:
    '''A 2:1 equirectangular sphere image ``(H, 2H)`` (W even, as the pole
    roll needs ``W/2``).'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((h, 2 * h)).astype(np.float32)


def _pad(img: Any, h_pad: int, w_pad: int, xp: Any) -> Any:
    '''Equirectangular pad on a 2D ``(H, W)`` image (default axes): wrap on W,
    pole-flip-and-roll on H (nitrix's convention).'''
    if w_pad > 0:
        mw = xp.concatenate(
            [img[:, -w_pad:], img, img[:, :w_pad]], axis=1)
    else:
        mw = img
    if h_pad > 0:
        roll = mw.shape[1] // 2
        top = xp.roll(mw[1:h_pad + 1][::-1], roll, axis=1)
        bot = xp.roll(mw[-(h_pad + 1):-1][::-1], roll, axis=1)
        mhw = xp.concatenate([top, mw, bot], axis=0)
    else:
        mhw = mw
    return mhw


def np_sphere_pad(pad: int) -> Callable[[Any], Any]:
    '''numpy equirectangular pad (CPU floor + fp64 oracle).'''

    def run(img: Any) -> Any:
        return _pad(np.asarray(img), pad, pad, np)

    return run


def np_sphere_unpad(pad: int) -> Callable[[Any], Any]:
    '''numpy unpad (inverse slice; CPU floor + fp64 oracle).'''

    def run(img: Any) -> Any:
        a = np.asarray(img)
        return a[pad:a.shape[0] - pad, pad:a.shape[1] - pad]

    return run


def cupy_sphere_pad(pad: int) -> Callable[[Any], Any]:
    '''GPU equirectangular pad (same wrap/pole-flip); cupy lazy.'''

    def run(img: Any) -> Any:
        import cupy as cp

        return _pad(img, pad, pad, cp)

    return run


def cupy_sphere_unpad(pad: int) -> Callable[[Any], Any]:
    '''GPU unpad (inverse slice); cupy lazy.'''

    def run(img: Any) -> Any:
        return img[pad:img.shape[0] - pad, pad:img.shape[1] - pad]

    return run
