# -*- coding: utf-8 -*-
"""Shared helpers for the IIR signal-filter family (sosfilt / sosfiltfilt).

These are recursive Butterworth IIR filters -- the **first non-embarrassingly-
parallel ops** in the suite: each output sample depends on the previous outputs
(a linear recurrence along time). nitrix offers two engines for the forward
filter: a sequential ``lax.scan`` (``backend='scan'``, O(T) depth) and a
parallel-prefix ``lax.associative_scan`` (``backend='associative'``, O(log T)
depth) -- the benchmark runs **both** so the sequential-vs-parallel tradeoff on
the GPU is explicit (the scan is ~70x slower on the L4).

scipy.signal is the canonical reference (nitrix matches it to ~1e-9 in nitrix's
own tests); CuPy's ``cupyx.scipy.signal`` is the on-target GPU reference. The
SOS coefficients are designed once with ``scipy.signal.butter`` (the canonical
design) and fed identically to nitrix / scipy / cupy, so the benchmark isolates
the recurrence **application**, not the (trivial) filter design.

scipy.signal is a core dep (top-level); cupy is lazy (refs-cupy worker only).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.signal as ss


def iir_input(channels: int, obs: int, seed: int = 0) -> np.ndarray:
    '''A (channels x obs) signal, filtered along the trailing (time) axis.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((channels, obs)).astype(np.float32)


def design_sos(order: int = 4, lo: float = 0.04, hi: float = 0.4,
               fs: float = 1.0) -> np.ndarray:
    '''Butterworth band-pass second-order sections (the canonical
    ``scipy.signal.butter`` design); fp32 so the recurrence runs in the same
    precision as the fp32 signal.'''
    return ss.butter(order, [lo, hi], btype='bandpass', output='sos',
                     fs=fs).astype(np.float32)


def sharp_sos(order: int = 8, band=(0.002, 0.004)) -> np.ndarray:
    '''A near-unstable narrow band-pass (poles hugging the unit circle,
    |pole| ~ 0.9996) whose impulse response does **not** decay below
    ``impulse_atol`` within the FFT engine's 2**15-tap cap -- so ``backend=
    'fft'`` falls back to the recurrence (with a warning).  This is the filter
    that exposes whether a bench is over-reporting the FFT win on filters where
    it does not apply (B18 Win 2): the realistic high-Q notch (Q<=200) is
    handled by the FFT path directly, but this one forces the fallback.'''
    return ss.butter(order, list(band), btype='bandpass',
                     output='sos').astype(np.float32)


def scipy_sosfilt(sos: np.ndarray, filtfilt: bool = False
                  ) -> Callable[[Any], Any]:
    '''scipy.signal forward (``sosfilt``) or zero-phase (``sosfiltfilt``)
    filter -- the CPU floor.'''
    fn = ss.sosfiltfilt if filtfilt else ss.sosfilt

    def run(x: Any) -> Any:
        return fn(sos, np.asarray(x), axis=-1)

    return run


def cupy_sosfilt(sos: np.ndarray, filtfilt: bool = False
                 ) -> Callable[[Any], Any]:
    '''CuPy ``cupyx.scipy.signal`` GPU reference; cupy lazy (refs-cupy env).'''

    def run(x: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.signal as css

        s = cp.asarray(sos)
        fn = css.sosfiltfilt if filtfilt else css.sosfilt
        return fn(s, x, axis=-1)

    return run
