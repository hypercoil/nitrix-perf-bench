# -*- coding: utf-8 -*-
"""Shared helpers for the augmentation family (``nitrix.augment``).

The augmentation ops are the medical-imaging perturbations of the FM
pretraining recipe. Where a deterministic core exists, MONAI -- the de-facto
community augmentation toolkit -- is the **community baseline** (torch-backed,
run CPU-only for now via the ``monai`` provider; the GPU headline ref is cupy).

This module carries, per op: a synthetic MRI-ish input, the numpy fp64 oracle
(nitrix's exact formula, the constants matched), a cupy GPU reference, and the
MONAI wrapper. cupy + monai are lazy-imported (their workers only).

**gamma_contrast.** nitrix's `gamma_contrast(x, gamma)` with the per-tensor
min/max bracket (`value_range=None`)::

    span = max(max(x) - min(x), eps);  normed = clip((x - lo) / span, 0, 1)
    out  = normed ** gamma * span + lo

equals MONAI `AdjustContrast(gamma)` (verified ~7e-8 in fp32 -- a clean
apples-to-apples community baseline). `eps = 1e-8` matches nitrix.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

_EPS = 1e-8  # matches nitrix.augment.intensity._EPS


def augment_input(shape, seed: int = 0) -> np.ndarray:
    '''A non-zero-mean, non-unit-scale MRI-ish volume (so the min/max bracket
    is non-trivial); gamma_contrast brackets over the full tensor.'''
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal(tuple(shape)) * 50.0 + 100.0
    return vol.astype(np.float32)


def _gamma(x: Any, xp: Any, gamma: float) -> Any:
    lo = xp.min(x)
    hi = xp.max(x)
    span = xp.maximum(hi - lo, xp.asarray(_EPS, dtype=x.dtype))
    normed = xp.clip((x - lo) / span, 0.0, 1.0)
    return normed ** gamma * span + lo


def np_gamma(gamma: float) -> Callable[[Any], np.ndarray]:
    '''nitrix's gamma-contrast formula in numpy -- the fp64 oracle.'''

    def run(x: Any) -> np.ndarray:
        return _gamma(np.asarray(x), np, gamma)

    return run


def cupy_gamma(gamma: float) -> Callable[[Any], Any]:
    '''cupy GPU reference (the same elementwise bracket+power; cupy lazy).'''

    def run(x: Any) -> Any:
        import cupy as cp

        return _gamma(x, cp, gamma)

    return run


def monai_adjust_contrast(gamma: float) -> Callable[[Any], Any]:
    '''MONAI ``AdjustContrast(gamma)`` -- the community baseline. Forced to a
    CPU torch tensor (the ``monai`` provider is CPU-only for now); returns the
    numpy result. monai/torch lazy-imported (the refs-monai worker only).'''

    def run(x: Any) -> Any:
        import torch
        from monai.transforms import AdjustContrast

        t = torch.from_numpy(np.ascontiguousarray(x)).to('cpu')
        out = AdjustContrast(gamma=gamma)(t)
        return np.asarray(out.detach().cpu().numpy())

    return run
