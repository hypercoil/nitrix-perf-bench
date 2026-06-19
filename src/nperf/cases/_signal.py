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


def narrowband_signal(n_sig: int, t: int, seed: int = 0) -> np.ndarray:
    '''A batch of **narrowband** signals -- per-signal linear chirps well below
    Nyquist, constant unit envelope -- the regime where the analytic-signal
    instantaneous phase/frequency are well-posed.

    White noise is pathological for ``instantaneous_frequency``: it has energy
    up to Nyquist, where the phase increment hits +-pi (the +pi == -pi wrap
    ambiguity) and the envelope dips to zero (undefined phase), so the result
    is unstable by NATURE -- not an fp32 or implementation defect (the
    conjugate-product reformulation does not help; verified).  See nitrix FR
    ``doc-instantaneous-frequency-narrowband-caveat``.  A chirp (the textbook
    instantaneous-frequency input) has a single well-defined frequency per
    sample, far from Nyquist, so fp32 matches the fp64 oracle.'''
    rng = np.random.default_rng(seed)
    tt = np.linspace(0.0, 1.0, t, dtype=np.float64)            # (t,)
    # start/end frequencies in CYCLES over the window; kept << Nyquist (= t/2
    # cycles), i.e. f/t << 0.5 cycles/sample even at the smallest t benched.
    f0 = rng.uniform(10.0, 30.0, (n_sig, 1))
    f1 = rng.uniform(40.0, 90.0, (n_sig, 1))
    phase = 2.0 * np.pi * (f0 * tt + 0.5 * (f1 - f0) * tt * tt)  # chirp phase
    sig = np.sin(phase) + 0.01 * rng.standard_normal((n_sig, t))  # tiny noise
    return sig.astype(np.float32)


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


# -- signal-extras family (instantaneous phase/freq, env_inst, product_filter)
import math  # noqa: E402

_PERIOD = 2.0 * math.pi


def freq_weight(t: int, seed: int = 0) -> np.ndarray:
    '''A real frequency-domain filter weight (rfft length ``t//2+1``): a smooth
    Gaussian lowpass taper -- the ``product_filter`` kernel.'''
    nf = t // 2 + 1
    f = np.linspace(0.0, 1.0, nf)
    return np.exp(-(f / 0.3) ** 2).astype(np.float32)


def _sp_hilbert(x: Any, fp64: bool) -> Any:
    import scipy.signal as ss
    return ss.hilbert(np.asarray(x, np.float64 if fp64 else np.float32),
                      axis=-1)


def scipy_inst_phase(fp64: bool = False) -> Callable[..., Any]:
    def run(x: Any) -> Any:
        return np.unwrap(np.angle(_sp_hilbert(x, fp64)), axis=-1)
    return run


def scipy_inst_freq(fp64: bool = False) -> Callable[..., Any]:
    def run(x: Any) -> Any:
        ph = np.unwrap(np.angle(_sp_hilbert(x, fp64)), axis=-1)
        return np.diff(ph, axis=-1) / _PERIOD       # fs = 1
    return run


def scipy_env_inst_sum(fp64: bool = False) -> Callable[..., Any]:
    '''``env_inst`` returns (envelope, inst_freq, inst_phase) from ONE analytic
    signal; we reduce to the scalar sum of all three to FORCE the full fused
    compute (no XLA dead-code elimination) + give a clean scalar fidelity.'''
    def run(x: Any) -> Any:
        z = _sp_hilbert(x, fp64)
        env = np.abs(z)
        ph = np.unwrap(np.angle(z), axis=-1)
        freq = np.diff(ph, axis=-1) / _PERIOD
        return np.asarray(env.sum() + freq.sum() + ph.sum())
    return run


def np_product_filter(fp64: bool = False) -> Callable[..., Any]:
    dt = np.float64 if fp64 else np.float32
    def run(x: Any, w: Any) -> Any:
        x = np.asarray(x, dt)
        w = np.asarray(w, dt)
        n = x.shape[-1]
        return np.fft.irfft(w * np.fft.rfft(x, n=n, axis=-1), n=n, axis=-1)
    return run


def np_product_filtfilt(fp64: bool = False) -> Callable[..., Any]:
    pf = np_product_filter(fp64)
    def run(x: Any, w: Any) -> Any:
        y = pf(x, w)
        y = pf(np.flip(y, axis=-1), w)
        return np.flip(y, axis=-1)
    return run


def cupy_env_inst_sum() -> Callable[..., Any]:
    def run(x: Any) -> Any:
        import cupy as cp
        from cupyx.scipy.signal import hilbert
        z = hilbert(x, axis=-1)
        env = cp.abs(z)
        ph = cp.unwrap(cp.angle(z), axis=-1)
        freq = cp.diff(ph, axis=-1) / _PERIOD
        return cp.asarray(env.sum() + freq.sum() + ph.sum())
    return run


def cupy_inst(kind: str) -> Callable[..., Any]:
    def run(x: Any) -> Any:
        import cupy as cp
        from cupyx.scipy.signal import hilbert
        ph = cp.unwrap(cp.angle(hilbert(x, axis=-1)), axis=-1)
        return cp.diff(ph, axis=-1) / _PERIOD if kind == 'freq' else ph
    return run


def cupy_product_filter(filtfilt: bool = False) -> Callable[..., Any]:
    def _pf(cp: Any, x: Any, w: Any) -> Any:
        n = x.shape[-1]
        return cp.fft.irfft(w * cp.fft.rfft(x, n=n, axis=-1), n=n, axis=-1)

    def run(x: Any, w: Any) -> Any:
        import cupy as cp
        y = _pf(cp, x, w)
        if filtfilt:
            y = cp.flip(_pf(cp, cp.flip(y, axis=-1), w), axis=-1)
        return y
    return run
