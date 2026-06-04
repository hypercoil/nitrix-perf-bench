# -*- coding: utf-8 -*-
"""Shared helpers for the numerics-normalize family.

``nitrix.numerics.{zscore_normalize, psc_normalize, robust_zscore_normalize,
intensity_normalize}`` are per-axis intensity normalisations (population
statistics, ``ddof=0``):

- **zscore**: ``(x - mean) / (std + eps)`` over the trailing axis -- equals
  ``scipy.stats.zscore`` (ddof=0 default; verified ~1e-12 in fp64, the eps),
  the canonical domain reference.
- **psc**: percent signal change ``100·(x - mean)/(mean + eps)`` (fMRI BOLD).
- **robust**: ``(x - median)/(1.4826·MAD + eps)``. nitrix uses the *truncated*
  literal ``1.4826``; ``scipy.stats.median_abs_deviation(scale='normal')`` uses
  the full ``1/Φ⁻¹(0.75) = 1.48260222``, so the two differ by ~1.5e-6 relative
  -- harmless (1.4826 is the conventional rounded value), and the oracle here
  matches nitrix's exact constant.
- **intensity**: percentile-clip to ``[p1, p99]`` then rescale to ``[0, 1]``
  (synthstrip / SynthSeg), over the whole tensor (``axis=None``).

All are memory-bound elementwise/reduction ops -> GPU-pure. scipy is a core
dep; cupy is lazy (its worker only). The numpy reimplementations carry nitrix's
``eps`` so they are exact fp64 oracles.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

_EPS = 1e-12
_MAD_SCALE = 1.4826  # nitrix's truncated Gaussian-consistency constant


def normalize_input(n: int, seed: int = 0) -> np.ndarray:
    '''A non-zero-mean, non-unit-scale matrix ``(n, n)`` (so the mean / std /
    percentile statistics are non-trivial); zscore/psc/robust reduce the
    trailing axis, intensity reduces the whole tensor.'''
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n)) * 3.0 + 2.0).astype(np.float32)


def _zscore(x: Any, xp: Any) -> Any:
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True)
    return (x - m) / (s + _EPS)


def _psc(x: Any, xp: Any) -> Any:
    m = x.mean(-1, keepdims=True)
    return 100.0 * (x - m) / (m + _EPS)


def _robust(x: Any, xp: Any) -> Any:
    med = xp.median(x, -1, keepdims=True)
    mad = xp.median(xp.abs(x - med), -1, keepdims=True)
    return (x - med) / (_MAD_SCALE * mad + _EPS)


def _intensity(x: Any, xp: Any) -> Any:
    lo = xp.percentile(x, 1.0)
    hi = xp.percentile(x, 99.0)
    return (xp.clip(x, lo, hi) - lo) / (hi - lo + _EPS)


# ---- numpy floors / fp64 oracles -----------------------------------------


def np_zscore(x: Any) -> np.ndarray:
    return _zscore(np.asarray(x), np)


def np_psc(x: Any) -> np.ndarray:
    return _psc(np.asarray(x), np)


def np_robust(x: Any) -> np.ndarray:
    return _robust(np.asarray(x), np)


def np_intensity(x: Any) -> np.ndarray:
    return _intensity(np.asarray(x), np)


def scipy_zscore() -> Callable[[Any], Any]:
    '''``scipy.stats.zscore`` (ddof=0) -- the canonical domain reference (CPU
    floor); no eps, so it differs from nitrix by ~1e-12.'''

    def run(x: Any) -> Any:
        from scipy import stats

        return stats.zscore(np.asarray(x), axis=-1)

    return run


# ---- cupy GPU references --------------------------------------------------


def _cupy(fn: Callable[[Any, Any], Any]) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        return fn(x, cp)

    return run


def cupy_zscore() -> Callable[[Any], Any]:
    return _cupy(_zscore)


def cupy_psc() -> Callable[[Any], Any]:
    return _cupy(_psc)


def cupy_robust() -> Callable[[Any], Any]:
    return _cupy(_robust)


def cupy_intensity() -> Callable[[Any], Any]:
    return _cupy(_intensity)
