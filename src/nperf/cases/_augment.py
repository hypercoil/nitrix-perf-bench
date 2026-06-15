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


# --- gibbs_ringing -------------------------------------------------------- #
# nitrix's gibbs: hard spherical k-space truncation -- zero every frequency
# whose normalised radius exceeds ``(1 - alpha)`` of the max radius, then
# inverse-FFT. NB MONAI ``GibbsNoise`` models a *different* artifact (a soft
# radial roll-off): it diverges ~0.18 here, so it is NOT a valid baseline (the
# signal-filter lesson). numpy is the fp64 oracle; cupy the GPU FFT headline.


def _gibbs(x: Any, xp: Any, alpha: float, axes=None) -> Any:
    x = xp.asarray(x)
    ndim = x.ndim
    fft_axes = tuple(range(ndim)) if axes is None else tuple(axes)
    grid_dtype = xp.result_type(x.dtype, xp.float32)
    r2 = xp.zeros((1,) * ndim, dtype=grid_dtype)
    for ax in fft_axes:
        n = x.shape[ax]
        coord = (xp.arange(n, dtype=grid_dtype) - n // 2) / max(n / 2.0, 1.0)
        shape = [1] * ndim
        shape[ax] = n
        r2 = r2 + coord.reshape(shape) ** 2
    radius = xp.sqrt(r2)
    mask = (radius <= (1.0 - alpha) * xp.max(radius)).astype(x.dtype)
    k = xp.fft.fftshift(xp.fft.fftn(x, axes=fft_axes), axes=fft_axes)
    k = k * mask
    out = xp.fft.ifftn(xp.fft.ifftshift(k, axes=fft_axes), axes=fft_axes)
    return out.real.astype(x.dtype)


def np_gibbs(alpha: float) -> Callable[[Any], np.ndarray]:
    '''nitrix's gibbs k-space truncation in numpy -- the fp64 oracle.'''

    def run(x: Any) -> np.ndarray:
        return _gibbs(np.asarray(x), np, alpha)

    return run


def cupy_gibbs(alpha: float) -> Callable[[Any], Any]:
    '''cupy GPU reference -- the on-target FFT headline bar (cupy lazy).'''

    def run(x: Any) -> Any:
        import cupy as cp

        return _gibbs(x, cp, alpha)

    return run


# --- gaussian_noise / rician_noise ---------------------------------------- #
# RNG ops: nitrix draws from jax PRNG, the refs from their own RNG, so there is
# NO cross-framework oracle (fp64_reference=None) -- the perf ratio is a fair
# task-level wall-clock comparison (generate N draws + combine, same sigma),
# and the distribution is validated in a test. cupy = GPU headline; MONAI
# RandGaussianNoise/RandRicianNoise = CPU community (matched: prob=1, mean=0,
# std=sigma, sample_std=False so the std is fixed, not itself sampled).


def np_gaussian_noise(sigma: float, seed: int) -> Callable[[Any], np.ndarray]:
    def run(x: Any) -> np.ndarray:
        x = np.asarray(x)
        rng = np.random.default_rng(seed)
        n = rng.standard_normal(x.shape, dtype=x.dtype)
        return x + np.asarray(sigma, x.dtype) * n

    return run


def cupy_gaussian_noise(sigma: float, seed: int) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        rng = cp.random.default_rng(seed)
        n = rng.standard_normal(x.shape, dtype=x.dtype)
        return x + cp.asarray(sigma, x.dtype) * n

    return run


def _np_rician(x: Any, sigma: float, seed: int, xp: Any, rng: Any) -> Any:
    n_r = rng.standard_normal(x.shape, dtype=x.dtype) * xp.asarray(sigma,
                                                                   x.dtype)
    n_i = rng.standard_normal(x.shape, dtype=x.dtype) * xp.asarray(sigma,
                                                                   x.dtype)
    return xp.sqrt((x + n_r) ** 2 + n_i ** 2)


def np_rician_noise(sigma: float, seed: int) -> Callable[[Any], np.ndarray]:
    def run(x: Any) -> np.ndarray:
        x = np.asarray(x)
        return _np_rician(x, sigma, seed, np, np.random.default_rng(seed))

    return run


def cupy_rician_noise(sigma: float, seed: int) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        return _np_rician(x, sigma, seed, cp, cp.random.default_rng(seed))

    return run


def _monai_noise(transform: str, sigma: float) -> Callable[[Any], Any]:
    '''MONAI Rand{Gaussian,Rician}Noise as a fixed-sigma community baseline
    (CPU torch tensor; prob=1, mean=0, std=sigma, sample_std=False).'''

    def run(x: Any) -> Any:
        import monai.transforms as MT
        import torch

        t = torch.from_numpy(np.ascontiguousarray(x)).to('cpu')
        if transform == 'gaussian':
            tf = MT.RandGaussianNoise(prob=1.0, mean=0.0, std=sigma,
                                      sample_std=False)
        else:
            tf = MT.RandRicianNoise(prob=1.0, mean=0.0, std=sigma,
                                    relative=False, sample_std=False)
        return np.asarray(tf(t).detach().cpu().numpy())

    return run


def monai_gaussian_noise(sigma: float) -> Callable[[Any], Any]:
    return _monai_noise('gaussian', sigma)


def monai_rician_noise(sigma: float) -> Callable[[Any], Any]:
    return _monai_noise('rician', sigma)


# --- random_flip / random_crop / random_histogram_shift / gmm ------------- #
# RNG ops: no cross-framework oracle (each framework draws its own randomness),
# so fp64_reference=None and the ratio is a task-level wall-clock comparison;
# correctness is pinned by a structural/distributional property test. cupy is
# the GPU headline; numpy the CPU floor; MONAI the community baseline where it
# maps (RandFlip / RandSpatialCrop / RandHistogramShift).


def _flip_axes(ndim: int, p: float, seed: int):
    rng = np.random.default_rng(seed)
    return tuple(ax for ax in range(ndim) if rng.random() < p)


def np_random_flip(p: float, seed: int) -> Callable[[Any], np.ndarray]:
    def run(x: Any) -> np.ndarray:
        x = np.asarray(x)
        ax = _flip_axes(x.ndim, p, seed)
        return np.ascontiguousarray(np.flip(x, ax)) if ax else x.copy()

    return run


def cupy_random_flip(p: float, seed: int) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        ax = _flip_axes(x.ndim, p, seed)  # axes on host; flip on device
        return cp.ascontiguousarray(cp.flip(x, ax)) if ax else x.copy()

    return run


def _crop_offset(shape, size, seed: int):
    rng = np.random.default_rng(seed)
    return tuple(int(rng.integers(0, s - sz + 1))
                 for s, sz in zip(shape, size))


def np_random_crop(size, seed: int) -> Callable[[Any], np.ndarray]:
    def run(x: Any) -> np.ndarray:
        x = np.asarray(x)
        off = _crop_offset(x.shape, size, seed)
        sl = tuple(slice(o, o + sz) for o, sz in zip(off, size))
        return np.ascontiguousarray(x[sl])  # materialise (a slice is a view)

    return run


def cupy_random_crop(size, seed: int) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        off = _crop_offset(x.shape, size, seed)
        sl = tuple(slice(o, o + sz) for o, sz in zip(off, size))
        return cp.ascontiguousarray(x[sl])

    return run


def _hist_control_points(lo: float, hi: float, n_cp: int, shift_range,
                         seed: int):
    '''The monotone remap table ``(ref, shifted)`` -- ``n_cp`` equally spaced
    reference levels, each perturbed by a random offset, endpoints pinned,
    cumulative-max for monotonicity (the nitrix / MONAI `RandHistogramShift`
    construction). Built on host: ``n_cp`` is tiny; the per-voxel interp is the
    work.'''
    rng = np.random.default_rng(seed)
    ref = np.linspace(lo, hi, n_cp)
    span = hi - lo
    shifted = ref + rng.uniform(shift_range[0], shift_range[1], n_cp) * span
    shifted[0], shifted[-1] = ref[0], ref[-1]      # pin endpoints
    shifted = np.maximum.accumulate(shifted)        # monotone (cummax)
    return ref, shifted


def np_random_histogram_shift(n_cp: int, shift_range,
                              seed: int) -> Callable[[Any], np.ndarray]:
    def run(x: Any) -> np.ndarray:
        x = np.asarray(x)
        ref, shifted = _hist_control_points(
            float(x.min()), float(x.max()), n_cp, shift_range, seed)
        return np.interp(x, ref, shifted).astype(x.dtype)

    return run


def cupy_random_histogram_shift(n_cp: int, shift_range,
                                seed: int) -> Callable[[Any], Any]:
    def run(x: Any) -> Any:
        import cupy as cp

        ref, shifted = _hist_control_points(
            float(x.min()), float(x.max()), n_cp, shift_range, seed)
        return cp.interp(x, cp.asarray(ref, x.dtype),
                         cp.asarray(shifted, x.dtype)).astype(x.dtype)

    return run


def monai_random_transform(kind: str, **kw) -> Callable[[Any], Any]:
    '''MONAI RandFlip / RandSpatialCrop / RandHistogramShift as community
    timing baselines (CPU torch, channel-first; prob=1 so it always applies).
    Not bit-identical to nitrix (different RNG / semantics) -- a task-level
    wall-clock community point, fidelity n/a.'''

    def run(x: Any) -> Any:
        import monai.transforms as MT
        import torch

        t = torch.from_numpy(np.ascontiguousarray(x))[None]  # (1, *spatial)
        if kind == 'flip':
            tf = MT.RandFlip(prob=1.0)
        elif kind == 'crop':
            tf = MT.RandSpatialCrop(roi_size=tuple(kw['size']),
                                    random_size=False)
        else:  # histogram_shift
            tf = MT.RandHistogramShift(num_control_points=kw['n_cp'], prob=1.0)
        out = np.asarray(tf(t).detach().cpu().numpy())
        return out[0]

    return run


def gmm_labels(shape, n_labels: int = 5, seed: int = 0):
    '''A blocky integer label map + per-label means/stds (means well above 0 so
    the nonneg clamp never bites -> per-label mean recovers the means).'''
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, n_labels, tuple(shape)).astype(np.int32)
    means = (10.0 + 10.0 * np.arange(n_labels)).astype(np.float32)
    stds = np.full(n_labels, 1.0, np.float32)
    return lab, means, stds


def np_gmm(means: Any, stds: Any, seed: int) -> Callable[[Any], np.ndarray]:
    def run(label_map: Any) -> np.ndarray:
        label_map = np.asarray(label_map)
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(label_map.shape, dtype=np.float32)
        m = np.asarray(means)[label_map]
        s = np.asarray(stds)[label_map]
        return np.maximum(m + s * noise, 0.0).astype(np.float32)  # nonneg

    return run


def cupy_gmm(means: Any, stds: Any, seed: int) -> Callable[[Any], Any]:
    def run(label_map: Any) -> Any:
        import cupy as cp

        rng = cp.random.default_rng(seed)
        noise = rng.standard_normal(label_map.shape, dtype=cp.float32)
        m = cp.asarray(np.asarray(means))[label_map]
        s = cp.asarray(np.asarray(stds))[label_map]
        return cp.maximum(m + s * noise, 0.0).astype(cp.float32)

    return run
