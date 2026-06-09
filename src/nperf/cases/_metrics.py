# -*- coding: utf-8 -*-
"""Shared helpers for the image-similarity metric family (registration R0).

``nitrix.metrics.{ssd, ncc, lncc, mutual_information, correlation_ratio}`` are
the similarity scores a registration loss is built from.  Each case benchmarks
nitrix against a **numpy exact-convention reimplementation** (the fp64 oracle +
CPU floor) and a **cupy** reimplementation (the GPU bar).  The domain tools are
added per metric, with the relationship nitrix documents (verified 2026-06-09):

- ``ssd`` (mean) == ITK ``MeanSquares`` bit-equal -> a **gated co-oracle**.
- ``ncc`` == signed Pearson ``r``; ITK ``Correlation`` returns ``-r**2`` -- a
  *different quantity* (squared, sign-dropped), so it rides as a **labelled
  divergent** ``ApproxBaseline`` (fidelity reported, not gated), not an oracle.
- ``lncc`` == ANTs ``ANTSNeighborhoodCorrelation`` on the interior; only the
  boundary trim differs -> ApproxBaseline (boundary divergence).
- ``mutual_information`` is an order-1 Parzen MI; ITK Mattes (order-3) and
  sklearn (order-0) are the *same family, different numbers* -> divergent refs.
- ``correlation_ratio`` (Roche eta^2): SimpleITK ships no CR metric -> numpy
  reimpl is the only reference.

The SimpleITK trap (DOMAIN_TOOL_BASELINES.md §7): ITK is ``(x, y, z)`` but
``GetImageFromArray`` consumes the reversed ``(z, y, x)`` numpy layout; the
metrics here are layout-symmetric (whole-image reductions), so the transpose is
immaterial to the value, but the images are still built via
``GetImageFromArray`` for correctness.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def metric_pair(shape, seed: int = 0,
                kind: str = 'within') -> Tuple[np.ndarray, np.ndarray]:
    '''A ``(moving, fixed)`` image pair with a realistic intensity relation --
    structured (smoothed noise), so the metric does genuine work (not a
    degenerate value).  ``within`` = linear intensity change + noise (the
    within-modality regime of ssd/ncc/lncc); ``cross`` = a non-monotonic remap
    (the cross-modal regime of MI/CR).'''
    import scipy.ndimage as spnd

    rng = np.random.default_rng(seed)
    base = spnd.gaussian_filter(
        rng.standard_normal(shape).astype(np.float32), sigma=2.0)
    base = (base - base.mean()) / (base.std() + 1e-6)
    noise = rng.standard_normal(shape).astype(np.float32)
    if kind == 'within':
        moving = 1.2 * base + 0.3 + 0.1 * noise
    else:  # cross-modal: a non-monotonic functional relationship
        moving = np.sin(2.0 * base).astype(np.float32) + 0.1 * noise
    return moving.astype(np.float32), base.astype(np.float32)


# --- numpy exact-convention reimplementations (fp64 oracle + CPU floor) ------

def np_ssd(moving: Any, fixed: Any) -> np.ndarray:
    '''Mean squared difference (nitrix default ``reduction='mean'`` == ITK
    MeanSquares).'''
    m = np.asarray(moving, np.float64)
    f = np.asarray(fixed, np.float64)
    return np.asarray(np.mean((m - f) ** 2))


def np_ncc(x: Any, y: Any) -> np.ndarray:
    '''Signed Pearson correlation over all voxels (nitrix ``ncc``).'''
    a = np.asarray(x, np.float64).ravel()
    b = np.asarray(y, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return np.asarray((a * b).sum() / (den + 1e-8))


# --- cupy reimplementations (the GPU bar; cupy lazy) -------------------------

def cupy_ssd() -> Callable[..., Any]:
    def run(moving: Any, fixed: Any) -> Any:
        import cupy as cp

        d = cp.asarray(moving) - cp.asarray(fixed)
        return cp.mean(d * d)

    return run


def cupy_ncc() -> Callable[..., Any]:
    def run(x: Any, y: Any) -> Any:
        import cupy as cp

        a = cp.asarray(x).ravel()
        b = cp.asarray(y).ravel()
        a = a - a.mean()
        b = b - b.mean()
        den = cp.sqrt((a * a).sum() * (b * b).sum())
        return (a * b).sum() / (den + 1e-8)

    return run


# --- SimpleITK domain references (host, base env; lazy) ----------------------

def _sitk_metric_eval(setter: str, *args: Any):
    '''A SimpleITK ``MetricEvaluate`` at identity: build fixed/moving images
    (``GetImageFromArray``), set the named metric, evaluate.  ``setter`` is the
    ``ImageRegistrationMethod`` method name (e.g. ``SetMetricAsMeanSquares``);
    ``args`` are forwarded to it (e.g. the Mattes bin count, the ANTs radius).
    ITK registration metrics return a *cost* (negated similarity); that, plus
    any binning/boundary difference, is what marks the divergent ones.'''

    def run(moving: Any, fixed: Any) -> float:
        import SimpleITK as sitk

        f = np.asarray(fixed, np.float64)
        m = np.asarray(moving, np.float64)
        reg = sitk.ImageRegistrationMethod()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetInitialTransform(sitk.TranslationTransform(f.ndim))
        getattr(reg, setter)(*args)
        return float(reg.MetricEvaluate(
            sitk.GetImageFromArray(f), sitk.GetImageFromArray(m)))

    return run


def sklearn_mi(bins: int = 32):
    '''Hard-binned (order-0) mutual information via sklearn -- the textbook MI,
    a *divergent* reference to nitrix's order-1 Parzen MI (a different number
    at a fixed bin count; natural-log nats, as nitrix).'''

    def run(moving: Any, fixed: Any) -> float:
        from sklearn.metrics import mutual_info_score

        m = np.asarray(moving, np.float64).ravel()
        f = np.asarray(fixed, np.float64).ravel()
        edges_m = np.linspace(m.min(), m.max(), bins + 1)[1:-1]
        edges_f = np.linspace(f.min(), f.max(), bins + 1)[1:-1]
        lm = np.clip(np.digitize(m, edges_m), 0, bins - 1)
        lf = np.clip(np.digitize(f, edges_f), 0, bins - 1)
        return float(mutual_info_score(lm, lf))

    return run


# --- numpy exact-convention reimplementations for lncc / MI / CR -------------

def _np_soft_bin(v: np.ndarray, bins: int, lo: float, hi: float):
    '''nitrix ``_soft_bin``: linear (Parzen) soft binning -> (lower, frac).'''
    span = max(hi - lo, 1e-12)
    scaled = np.clip((v - lo) / span * (bins - 1), 0.0, bins - 1)
    lower = np.clip(np.floor(scaled).astype(np.int64), 0, bins - 2)
    return lower, scaled - lower


def _np_box_sum(x: np.ndarray, size: int, mode: str = 'reflect') -> np.ndarray:
    '''nitrix ``_box_sum``: separable windowed sum (ones-kernel correlate1d).
    scipy ``mode='reflect'`` matches nitrix's reflect boundary (verified).'''
    import scipy.ndimage as spnd

    out = x
    k = np.ones(size)
    for ax in range(x.ndim):
        out = spnd.correlate1d(out, k, axis=ax, mode=mode)
    return out


def np_lncc(moving: Any, fixed: Any, radius: int = 4,
            eps: float = 1e-5) -> np.ndarray:
    '''ANTs squared local CC over a box window (nitrix ``lncc``, reflect).'''
    m = np.asarray(moving, np.float64)
    f = np.asarray(fixed, np.float64)
    size = 2 * radius + 1
    n = size ** m.ndim
    sm, sf = _np_box_sum(m, size), _np_box_sum(f, size)
    smm, sff = _np_box_sum(m * m, size), _np_box_sum(f * f, size)
    smf = _np_box_sum(m * f, size)
    cross = smf - sm * sf / n
    var_m = smm - sm * sm / n
    var_f = sff - sf * sf / n
    return np.asarray(np.mean(cross ** 2 / (var_m * var_f + eps)))


def _np_joint_hist(m: np.ndarray, f: np.ndarray, bins: int) -> np.ndarray:
    lm, fm = _np_soft_bin(m, bins, m.min(), m.max())
    lf, ff = _np_soft_bin(f, bins, f.min(), f.max())
    h = np.zeros((bins, bins))
    for im, wm in ((lm, 1 - fm), (lm + 1, fm)):
        for jf, wf in ((lf, 1 - ff), (lf + 1, ff)):
            np.add.at(h, (im, jf), wm * wf)
    return h / max(h.sum(), 1e-12)


def np_mi(moving: Any, fixed: Any, bins: int = 32,
          eps: float = 1e-10) -> np.ndarray:
    '''Order-1 Parzen mutual information (nitrix ``mutual_information``).'''
    m = np.asarray(moving, np.float64).ravel()
    f = np.asarray(fixed, np.float64).ravel()
    h = _np_joint_hist(m, f, bins)
    p_m, p_f = h.sum(1), h.sum(0)
    outer = p_m[:, None] * p_f[None, :]
    ratio = np.where(h > 0, h / (outer + eps), 1.0)
    return np.asarray(np.sum(np.where(h > 0, h * np.log(ratio), 0.0)))


def np_cr(moving: Any, fixed: Any, bins: int = 32,
          eps: float = 1e-10) -> np.ndarray:
    '''Roche correlation ratio eta^2 (nitrix ``correlation_ratio``).'''
    m = np.asarray(moving, np.float64).ravel()
    f = np.asarray(fixed, np.float64).ravel()
    lf, ff = _np_soft_bin(f, bins, f.min(), f.max())
    n_k, s_k = np.zeros(bins), np.zeros(bins)
    for idx, w in ((lf, 1 - ff), (lf + 1, ff)):
        np.add.at(n_k, idx, w)
        np.add.at(s_k, idx, w * m)
    mu = m.mean()
    mu_k = s_k / (n_k + eps)
    between = np.sum(n_k * (mu_k - mu) ** 2)
    total = np.sum((m - mu) ** 2)
    return np.asarray(between / (total + eps))


# --- cupy reimplementations for lncc / MI / CR (the GPU bar; lazy) -----------

def cupy_lncc(radius: int = 4) -> Callable[..., Any]:
    def run(moving: Any, fixed: Any) -> Any:
        import cupy as cp
        from cupyx.scipy.ndimage import correlate1d

        m = cp.asarray(moving, cp.float64)
        f = cp.asarray(fixed, cp.float64)
        size = 2 * radius + 1
        n = size ** m.ndim

        def bs(x: Any) -> Any:
            out = x
            k = cp.ones(size)
            for ax in range(x.ndim):
                out = correlate1d(out, k, axis=ax, mode='reflect')
            return out

        sm, sf = bs(m), bs(f)
        cross = bs(m * f) - sm * sf / n
        var_m = bs(m * m) - sm * sm / n
        var_f = bs(f * f) - sf * sf / n
        return cp.mean(cross ** 2 / (var_m * var_f + 1e-5))

    return run


def _cupy_soft_bin(v: Any, bins: int, cp: Any):
    span = cp.maximum(v.max() - v.min(), 1e-12)
    scaled = cp.clip((v - v.min()) / span * (bins - 1), 0.0, bins - 1)
    lower = cp.clip(cp.floor(scaled).astype(cp.int64), 0, bins - 2)
    return lower, scaled - lower


def cupy_mi(bins: int = 32) -> Callable[..., Any]:
    def run(moving: Any, fixed: Any) -> Any:
        import cupy as cp

        m = cp.asarray(moving).ravel()
        f = cp.asarray(fixed).ravel()
        lm, fm = _cupy_soft_bin(m, bins, cp)
        lf, ff = _cupy_soft_bin(f, bins, cp)
        h = cp.zeros(bins * bins)
        for im, wm in ((lm, 1 - fm), (lm + 1, fm)):
            for jf, wf in ((lf, 1 - ff), (lf + 1, ff)):
                h += cp.bincount(im * bins + jf, weights=wm * wf,
                                 minlength=bins * bins)
        h = (h / cp.maximum(h.sum(), 1e-12)).reshape(bins, bins)
        p_m, p_f = h.sum(1), h.sum(0)
        outer = p_m[:, None] * p_f[None, :]
        ratio = cp.where(h > 0, h / (outer + 1e-10), 1.0)
        return cp.sum(cp.where(h > 0, h * cp.log(ratio), 0.0))

    return run


def cupy_cr(bins: int = 32) -> Callable[..., Any]:
    def run(moving: Any, fixed: Any) -> Any:
        import cupy as cp

        m = cp.asarray(moving).ravel()
        f = cp.asarray(fixed).ravel()
        lf, ff = _cupy_soft_bin(f, bins, cp)
        n_k = cp.zeros(bins)
        s_k = cp.zeros(bins)
        for idx, w in ((lf, 1 - ff), (lf + 1, ff)):
            n_k = n_k + cp.bincount(idx, weights=w, minlength=bins)
            s_k = s_k + cp.bincount(idx, weights=w * m, minlength=bins)
        mu = m.mean()
        mu_k = s_k / (n_k + 1e-10)
        between = cp.sum(n_k * (mu_k - mu) ** 2)
        total = cp.sum((m - mu) ** 2)
        return between / (total + 1e-10)

    return run
