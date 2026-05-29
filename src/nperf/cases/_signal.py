# -*- coding: utf-8 -*-
"""Shared helpers for the Hilbert / analytic-signal cases (Tier-2).

``analytic_signal`` / ``hilbert_transform`` / ``envelope`` are all the
FFT-based analytic signal (Hilbert transform along the last axis): nitrix vs
``scipy.signal.hilbert`` (CPU floor) + ``cupyx.scipy.signal.hilbert`` (GPU
ref), scored against an fp64 scipy oracle.  Being FFT-based, they run
**GPU-pure** (no cuSolver), so -- unlike the eigh family -- they give a clean
apples-to-apples GPU bar at every size.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def signal_input(n_sig: int, t: int, seed: int = 0) -> np.ndarray:
    '''A batch of real time series: ``n_sig`` signals x ``t`` samples.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_sig, t)).astype(np.float32)


def cupy_hilbert(kind: str) -> Callable[[Any], Any]:
    '''GPU Hilbert (``cupyx.scipy.signal.hilbert``) along the last axis; cupy
    imported lazily.  ``kind``: ``complex`` (analytic signal) | ``imag`` (the
    Hilbert transform) | ``abs`` (the envelope).'''

    def run(x: Any) -> Any:
        import cupy as cp
        from cupyx.scipy.signal import hilbert

        z = hilbert(x, axis=-1)
        if kind == 'complex':
            return z
        if kind == 'imag':
            return z.imag
        return cp.abs(z)

    return run
