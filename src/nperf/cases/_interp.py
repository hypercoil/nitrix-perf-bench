# -*- coding: utf-8 -*-
"""Shared helpers for ``nitrix.signal.lomb_scargle_interpolate`` (Power 2014).

Joint-GLM interpolation of motion-censored fMRI: fit the observed (non-masked)
samples to a ``[DC | cos | sin]`` Lomb-Scargle basis by **masked** least
squares (an eigh-truncated pseudo-inverse of the masked Gram), then **splice**
the observed samples through unchanged (``where(mask, data, recon)``).  The
joint fit -- unlike the per-frequency periodogram reconstruction -- passes
through the observed samples exactly, so the spliced output has no boundary
discontinuity (the whole point; see ``signal-and-numerics.md``).

Reference: a from-scratch joint-GLM (numpy / cupy) that replicates this
algorithm (NOT ``astropy``'s ``LombScargle.model()`` -- the per-frequency
reconstruction whose boundary jumps nitrix's rewrite fixed).  The trial-
frequency grid is reproduced from nitrix's Press-Rybicki convention (verified
against ``nitrix.signal.lomb_scargle._trial_frequencies`` in the tests).

**No cross-impl fidelity oracle (``fp64_reference=None``).** The masked Gram is
hugely ill-conditioned (cond ~1e32); the rcond-truncated pseudo-inverse
regularises it, but *which* near-zero eigenvalues fall below the truncation
threshold differs between fp32 and fp64 (and between nitrix's fp32 grid and a
fp64 grid) -- so the **censored-frame reconstruction has no well-defined
bit-level truth** (measured: worst-frame fp32-vs-fp64 disagreement ~1.4 on
O(1) signals).  This is the DESIGN no-cross-impl-oracle case (SCHEMA §C):
fidelity is recorded *inconclusive*, the ratio is still emitted.  The
**load-bearing correctness -- observed-sample splice-through
(``recon[obs] == data[obs]`` exactly) -- IS well-defined and is asserted in the
tests** instead of via the fidelity gate.

GPU: nitrix routes the K x K Gram eigh through ``safe_eigh`` -> CPU on this L4
(eigh-family; see [[perfbench-gpu-eigh-blocker]]), the batched solve stays on
device -- so nitrix runs on GPU (hybrid).  The CuPy reference's
``cupy.linalg.eigh`` hits cuSolver and fails at K>=256 (recorded
``gpu_solver_unavailable``), so the apples-to-apples GPU bar holds at small
K (small ``obs``).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

_DT = 1.0
_OVERSAMPLING = 4.0
_HIGH_FACTOR = 1.0
_RCOND = 1e-6
_CENSORING_BUDGET = 0.4


def interp_input(
    n_vox: int, obs: int, seed: int = 0, censor_frac: float = 0.15,
) -> tuple:
    '''Batch of time series ``data (V, obs)`` + a **shared** censor ``mask
    (obs,)`` (~``censor_frac`` of frames dropped) -- the canonical fMRI
    motion-censoring case (one mask per scan, applied to all voxels).'''
    rng = np.random.default_rng(seed)
    t = np.arange(obs)
    base = (np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.11 * t))
    data = (base + 0.4 * rng.standard_normal((n_vox, obs))).astype(np.float32)
    mask = rng.random(obs) > censor_frac
    return data, mask


def _omega(obs: int, dtype: Any) -> np.ndarray:
    '''Press-Rybicki trial angular frequencies (matches nitrix's
    ``_trial_frequencies``; verified in the tests).'''
    big_t = obs * _DT
    df = 1.0 / (big_t * _OVERSAMPLING)
    f_max = _HIGH_FACTOR / (2.0 * _DT)
    n_grid = max(int(f_max / df), 1)
    margin = max(int(obs * (1.0 - _CENSORING_BUDGET)), 3)
    n_cap = max((margin - 1) // 2, 1)
    n_freq = min(n_grid, n_cap)
    return 2.0 * np.pi * np.arange(1, n_freq + 1, dtype=dtype) * df


def joint_glm(data: Any, mask: Any, xp: Any) -> Any:
    '''Masked-LS joint-GLM interpolation via ``xp`` (numpy or cupy), in
    ``data``'s dtype.  ``data`` is ``(V, obs)``, ``mask`` is ``(obs,)``.'''
    obs = data.shape[-1]
    omega = xp.asarray(_omega(obs, np.float64).astype(data.dtype))
    t = xp.arange(obs, dtype=data.dtype) * _DT
    arg = omega[None, :] * t[:, None]
    basis = xp.concatenate(
        [xp.ones((obs, 1), data.dtype), xp.cos(arg), xp.sin(arg)], axis=1)
    mf = mask.astype(data.dtype)
    b_w = basis * mf[:, None]                       # (obs, K)
    gram = b_w.T @ basis                            # (K, K) symmetric PSD
    ev, vec = xp.linalg.eigh(gram)                  # ascending
    ev_inv = xp.where(ev > _RCOND * ev[-1], 1.0 / ev, 0.0)
    rhs = data @ b_w                                # (V, K)
    beta = vec @ (ev_inv[:, None] * (vec.T @ rhs.T))  # (K, V)
    recon = (basis @ beta).T                        # (V, obs)
    return xp.where(mask[None, :], data, recon)


def cupy_joint_glm() -> Callable[[Any, Any], Any]:
    '''CuPy GPU joint-GLM (cupy lazy; refs-cupy env).  ``cupy.linalg.eigh``
    hits cuSolver -> fails at K>=256.'''

    def run(data: Any, mask: Any) -> Any:
        import cupy as cp

        return joint_glm(data, mask, cp)

    return run
