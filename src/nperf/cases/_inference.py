# -*- coding: utf-8 -*-
"""Shared generators + community baselines for the permutation-inference family
(``nitrix.stats.inference``): the supporting ops (TFCE enhancement, cluster
size/mass maps, FDR/Bonferroni multiple-comparison correction, GPD-tail
p-values) and the headline ``permutation_test`` (a separate module).

Baselines: **statsmodels** ``multipletests`` (the canonical FDR/Bonferroni),
**scipy** (``ndimage.label`` for clusters, ``stats.genpareto`` for the GPD
tail), and exact **numpy** reimplementations as fp64 oracles.  GPU bars where a
``cupyx.scipy.ndimage`` twin exists (TFCE / clusters).
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def stat_map(shape: Tuple[int, ...], seed: int = 0) -> np.ndarray:
    '''A signed statistic image with a few bright blobs (so thresholding gives
    real clusters): smooth gaussian bumps + noise.'''
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(np.float32)
    # add a few high-intensity blobs
    from scipy.ndimage import gaussian_filter
    blobs = rng.standard_normal(shape).astype(np.float32)
    smooth = gaussian_filter(blobs, sigma=2.0) * 12.0
    return (base + smooth).astype(np.float32)


def labels_from(stat: np.ndarray, threshold: float, conn: int = 1
                ) -> np.ndarray:
    '''Connected-component labels of ``stat > threshold`` (scipy.ndimage, the
    cluster_*_map input) -- generated host-side so both nitrix and the oracle
    label the SAME clusters.'''
    from scipy.ndimage import generate_binary_structure, label
    st = generate_binary_structure(stat.ndim, conn)
    lab, _ = label(stat > threshold, st)
    return lab.astype(np.int32)


def pvalues(n: int, seed: int = 0) -> np.ndarray:
    '''A vector of p-values: mostly uniform null + a sprinkle of small (signal)
    -- the multiple-comparison input.'''
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    p[: n // 20] = rng.uniform(0.0, 0.01, n // 20)  # 5% "signal"
    return p.astype(np.float32)


def null_dist(n: int, seed: int = 0) -> np.ndarray:
    '''A max-statistic null distribution (the GPD-tail input).'''
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(n)).astype(np.float32) * 1.5


# -- numpy fp64 oracles ------------------------------------------------------
def np_tfce(E: float = 0.5, H: float = 2.0, n_steps: int = 100,
            conn: int = 1) -> Callable[..., Any]:
    '''Threshold-free cluster enhancement (one-sided): integrate
    ``extent(h)^E * h^H`` over ``n_steps`` thresholds (Smith-Nichols).'''
    from scipy.ndimage import generate_binary_structure, label

    def run(stat: Any) -> Any:
        s = np.asarray(stat, np.float64)
        st = generate_binary_structure(s.ndim, conn)
        hmax = float(s.max())
        dh = hmax / n_steps
        out = np.zeros_like(s)
        for i in range(1, n_steps + 1):
            h = i * dh
            lab, k = label(s >= h, st)
            if k == 0:
                continue
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            out += (sizes[lab] ** E) * (h ** H) * dh
        return out
    return run


def np_cluster_size() -> Callable[..., Any]:
    def run(labels: Any) -> Any:
        lab = np.asarray(labels)
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        return counts[lab]
    return run


def np_cluster_mass(threshold: float) -> Callable[..., Any]:
    def run(labels: Any, stat: Any) -> Any:
        lab = np.asarray(labels)
        s = np.asarray(stat, np.float64)
        excess = np.where(lab > 0, s - threshold, 0.0)
        sums = np.bincount(lab.ravel(), weights=excess.ravel())
        sums[0] = 0.0
        return sums[lab]
    return run


def sm_multipletests(method: str, alpha: float = 0.05
                     ) -> Callable[..., Any]:
    '''statsmodels multiple-comparison correction (the canonical impl) ->
    corrected p-values.'''
    def run(p: Any) -> Any:
        from statsmodels.stats.multitest import multipletests
        return multipletests(np.asarray(p, np.float64), alpha=alpha,
                             method=method)[1]
    return run


def np_fdr_bh(alpha: float = 0.05) -> Callable[..., Any]:
    def run(p: Any) -> Any:
        p = np.asarray(p, np.float64)
        n = p.size
        order = np.argsort(p)
        ranked = p[order] * n / (np.arange(n) + 1)
        padj = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.clip(padj, 0, 1)
        return out
    return run


def np_bonferroni(alpha: float = 0.05) -> Callable[..., Any]:
    def run(p: Any) -> Any:
        p = np.asarray(p, np.float64)
        return np.clip(p * p.size, 0, 1)
    return run


def scipy_gpd(n_exc: int = 250) -> Callable[..., Any]:
    '''scipy genpareto-MLE tail p-value (the Knijnenburg GPD approximation) --
    a CLOSE community cross-check (the GPD fit method differs from nitrix, so
    ~1-2% on the tail p; gated loosely).'''
    def run(stat: Any, null: Any) -> Any:
        from scipy.stats import genpareto
        null = np.asarray(null, np.float64)
        stat = np.asarray(stat, np.float64)
        thr = np.sort(null)[-n_exc - 1]
        exc = null[null > thr] - thr
        c, _, scale = genpareto.fit(exc, floc=0)
        return (exc.size / null.size) * genpareto.sf(stat - thr, c, loc=0,
                                                     scale=scale)
    return run
