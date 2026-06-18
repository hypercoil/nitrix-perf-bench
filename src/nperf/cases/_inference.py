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
            lab, k = label(s > h, st)  # STRICT '>' (matches nitrix's ladder)
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


def np_gpd(n_exceedances: int = 250) -> Callable[..., Any]:
    '''Exact numpy fp64 reimplementation of nitrix ``gpd_pvalue`` (the oracle):
    Winkler-2016 tail acceleration with a **method-of-moments** GPD above the
    (k+1)-th-largest threshold ``u`` and the **empirical** survival below it::

        P(T) = (k/n) * S_GPD(T - u),  T > u    (MoM: xi=(1-ybar^2/var)/2,
        P(T) = #{null >= T} / n,      T <= u                sigma=ybar*(1-xi))
    '''
    def run(stat: Any, null: Any) -> Any:
        s = np.asarray(stat, np.float64)
        nl = np.sort(np.asarray(null, np.float64))  # ascending
        n = nl.shape[0]
        k = min(int(n_exceedances), n - 1)
        u = nl[n - k - 1]                      # (k+1)-th largest
        exc = nl[n - k:] - u                   # top-k exceedances
        ybar = exc.mean()
        s2 = exc.var()                         # population variance (ddof=0)
        xi = 0.5 * (1.0 - ybar * ybar / np.clip(s2, 1e-30, None))
        sigma = np.clip(ybar * (1.0 - xi), 1e-30, None)
        t = s - u
        z = 1.0 + xi * t / sigma
        surv = np.where(np.abs(xi) < 1e-6, np.exp(-t / sigma),
                        np.clip(z, 0.0, None) ** (-1.0 / xi))
        p_tail = np.clip((k / n) * surv, 0.0, 1.0)
        idx = np.searchsorted(nl, s, side='left')
        p_emp = (n - idx) / n                  # empirical body
        return np.where(s > u, p_tail, p_emp)
    return run


def scipy_gpd(n_exc: int = 250) -> Callable[..., Any]:
    '''scipy genpareto-**MLE** tail p-value -- a different-fit community cross-
    check (nitrix uses method-of-moments; an ApproxBaseline, NOT the oracle).
    Uses the SAME empirical body below the threshold as nitrix (so the body is
    bit-identical), and only the GPD tail diverges -- MEASURED up to ~10-15% at
    the most extreme statistics (where the GPD extrapolates furthest and the
    fit method matters most), exact at moderate tail values.'''
    def run(stat: Any, null: Any) -> Any:
        from scipy.stats import genpareto
        nl = np.sort(np.asarray(null, np.float64))
        stat = np.asarray(stat, np.float64)
        n = nl.shape[0]
        k = min(int(n_exc), n - 1)
        u = nl[n - k - 1]
        exc = nl[n - k:] - u
        c, _, scale = genpareto.fit(exc, floc=0)        # MLE (vs nitrix MoM)
        p_tail = np.clip((k / n) * genpareto.sf(stat - u, c, loc=0,
                                                scale=scale), 0.0, 1.0)
        idx = np.searchsorted(nl, stat, side='left')
        p_emp = (n - idx) / n
        return np.where(stat > u, p_tail, p_emp)
    return run


# -- cupy GPU community bars (cupyx.scipy.ndimage.label) ----------------------
# TFCE and the cluster maps are connected-component ops; cupyx.scipy.ndimage
# gives the GPU community reference nitrix's jitted device path competes with.
# All use connectivity-1 (structure=None == generate_binary_structure(ndim,1)),
# matching the nitrix default ``connectivity=1`` and the numpy oracles above.
def cupy_tfce(E: float = 0.5, H: float = 2.0, n_steps: int = 100
              ) -> Callable[..., Any]:
    '''One-sided TFCE on the GPU (mirrors ``np_tfce``); compare to nitrix
    ``tfce(..., two_sided=False)``.'''
    def run(stat: Any) -> Any:
        import cupy as cp
        from cupyx.scipy.ndimage import label
        s = cp.asarray(stat, cp.float32)
        hmax = float(s.max())
        dh = hmax / n_steps
        out = cp.zeros_like(s)
        for i in range(1, n_steps + 1):
            h = i * dh
            lab, k = label(s > h)  # STRICT '>' (matches nitrix's ladder)
            if k == 0:
                continue
            sizes = cp.bincount(lab.ravel())
            sizes[0] = 0
            out += (sizes[lab].astype(cp.float32) ** E) * (h ** H) * dh
        return out
    return run


def cupy_cluster_size() -> Callable[..., Any]:
    def run(labels: Any) -> Any:
        import cupy as cp
        lab = cp.asarray(labels)
        counts = cp.bincount(lab.ravel()).astype(cp.float32)
        counts[0] = 0
        return counts[lab]
    return run


def cupy_cluster_mass(threshold: float) -> Callable[..., Any]:
    def run(labels: Any, stat: Any) -> Any:
        import cupy as cp
        lab = cp.asarray(labels)
        s = cp.asarray(stat, cp.float64)
        excess = cp.where(lab > 0, s - threshold, 0.0).ravel()
        sums = cp.bincount(lab.ravel(), weights=excess)
        sums[0] = 0.0
        return sums[lab]
    return run


# -- FSL community BUNDLE for the cluster-extent CHAIN ------------------------
# No standalone community tool computes nitrix's `cluster_size_map` /
# `connected_components` in isolation -- FSL FUSES threshold + CC-labelling +
# size into one pass.  So the fair comparison is the nitrix CHAIN
# `cluster_size_map(supra_threshold_clusters(stat, thr))` (one jit) vs FSL
# `fsl-cluster --osize` (the bundle).  --connectivity=6 == nitrix
# connectivity=1 == scipy 6-neighbour.  The size map is label-permutation-
# invariant, so it is directly comparable to the numpy oracle.
def np_cluster_extent(threshold: float, conn: int = 1) -> Callable[..., Any]:
    '''Oracle for the cluster-extent CHAIN: scipy label(stat>thr) -> size map
    (the same fused threshold+CC+size FSL does), in fp64.'''
    def run(stat: Any) -> Any:
        lab = labels_from(np.asarray(stat), threshold, conn)
        return np_cluster_size()(lab)
    return run


def fsl_cluster_osize(threshold: float, connectivity: int = 6
                      ) -> Callable[..., Any]:
    '''FSL `fsl-cluster --in --thresh --osize` -- the upstream cluster bundle:
    threshold -> CC label -> per-voxel size image, in ONE call. The fair
    competitor for the nitrix extent chain. Binary at `$NPERF_FSL_DIR/bin`
    (default `/scratch/nperf/fsl`); CPU-only, not jit (full wall-clock).
    --connectivity=6 matches nitrix connectivity=1.'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(stat: Any) -> Any:
        import subprocess
        import tempfile

        import nibabel as nib

        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        s = np.asarray(stat, np.float32)
        with tempfile.TemporaryDirectory(
                dir=os.environ.get('TMPDIR')) as d:
            nib.save(nib.Nifti1Image(s, np.eye(4)), f'{d}/stat.nii.gz')
            subprocess.run(
                [os.path.join(fsldir, 'bin', 'fsl-cluster'),
                 '--in=stat.nii.gz', f'--thresh={threshold}',
                 '--osize=size', f'--connectivity={connectivity}',
                 '--no_table'],
                cwd=d, check=True, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return np.asarray(nib.load(f'{d}/size.nii.gz').get_fdata(),
                              np.float64).reshape(s.shape)

    return run


def fsl_cluster_iofloor() -> Callable[..., Any]:
    '''I/O floor for `fsl_cluster_osize`: the same stat NIfTI write + a
    subprocess (`fslmaths -mul 1`) + read-back, but NO clustering -- the
    file-coupling artifact nitrix never pays (economic_report subtracts the
    same-namespace `fsl.iofloor`).'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(stat: Any) -> Any:
        import subprocess
        import tempfile

        import nibabel as nib

        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        s = np.asarray(stat, np.float32)
        with tempfile.TemporaryDirectory(
                dir=os.environ.get('TMPDIR')) as d:
            nib.save(nib.Nifti1Image(s, np.eye(4)), f'{d}/stat.nii.gz')
            subprocess.run(
                [os.path.join(fsldir, 'bin', 'fslmaths'), 'stat.nii.gz',
                 '-mul', '1', 'floorout.nii.gz'],
                cwd=d, check=True, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _ = np.asarray(nib.load(f'{d}/floorout.nii.gz').get_fdata())
        return np.zeros(s.shape, np.float64)  # floor: timing only, not scored

    return run
