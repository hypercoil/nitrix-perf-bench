# -*- coding: utf-8 -*-
"""Real group-fMRI first-level estimates for the LME cases (vs synthetic).

The marquee real-data bar for ``flame_two_level`` (and the LME family): a
**real** two-level group analysis instead of the planted balanced design.
Source: **nilearn**'s Brainomics localizer (``fetch_localizer_contrasts``) --
per-subject first-level **effect** (COPE) and **t** maps for a motor contrast,
from which the within-subject variance is ``VARCOPE = (COPE / t)^2`` (the exact
identity ``t = COPE / sqrt(VARCOPE)``).  This is the actual input FSL FLAME
consumes: per-subject level-1 effects + their variances + a group design.

**Why this is ``real_full`` (no oracle).**  The synthetic ``flame_input`` uses
a *constant* known within-variance so the single-parameter REML has a closed
form (``flame_closed_form``) -- a real oracle.  Real VARCOPEs are
**heterogeneous**
across subjects and voxels, so that closed form no longer holds: there is no
cross-implementation oracle.  The validation is **agreement with FSL FLAME**
(``flameo``) -- the upstream gold standard nitrix targets -- on the real data
(asserted opt-in in the tests + read each authoritative sweep).

**Env / reproducibility.**  The assembled ``(beta, varw)`` arrays are cached as
a plain ``.npz`` under ``$NPERF_REAL_DATA`` (default
``/scratch/nperf/real_anatomy``): the first call materialises them via nilearn
(a one-time OSF download), every later call -- including the **FSL /
statsmodels ref workers, which have no nilearn** -- reads the numpy cache.
``/scratch`` is
ephemeral, so it regenerates on demand; pre-warm once in the base env before a
ref-env sweep.  ``nilearn`` / ``nibabel`` are imported only on a cache miss.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np

_CACHE = os.environ.get('NPERF_REAL_DATA', '/scratch/nperf/real_anatomy')
# A robust motor contrast present for every localizer subject.
_CONTRAST = 'left vs right button press'
_TFLOOR = 1e-3  # |t| floor: a voxel needs a well-defined VARCOPE = (cope/t)^2
# reml's random-effect grouping: acquisition SITE -- the canonical neuroimaging
# random factor (multi-site studies model site as a random intercept).
_REML_GROUP = 'site'


def _materialise(n_subjects: int, n_vox: int, seed: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    '''Fetch the localizer COPE+t maps and assemble ``(beta (V, N),
    varw (V, N))`` over a reproducible in-brain voxel sample.  nilearn /
    nibabel imported here only (the cache-miss path).'''
    import nibabel as nib
    from nilearn import datasets as ds

    os.makedirs(_CACHE, exist_ok=True)
    dat = ds.fetch_localizer_contrasts(
        [_CONTRAST], n_subjects=n_subjects, get_tmaps=True, data_dir=_CACHE)
    cmaps, tmaps = dat['cmaps'], dat['tmaps']
    eff = np.stack([np.asarray(nib.load(c).get_fdata(), np.float64)
                    for c in cmaps], -1)              # (X, Y, Z, N)
    tval = np.stack([np.asarray(nib.load(t).get_fdata(), np.float64)
                     for t in tmaps], -1)
    eff = eff.reshape(-1, eff.shape[-1])               # (vox, N)
    tval = tval.reshape(-1, tval.shape[-1])
    # in-brain voxels: finite and a well-defined variance for EVERY subject.
    ok = (np.isfinite(eff).all(1) & np.isfinite(tval).all(1)
          & (np.abs(tval) > _TFLOOR).all(1))
    idx = np.flatnonzero(ok)
    if idx.size < n_vox:
        raise RuntimeError(
            f'localizer: only {idx.size} usable voxels < requested {n_vox}')
    pick = np.random.default_rng(seed).choice(idx, n_vox, replace=False)
    pick.sort()
    beta = eff[pick]                                   # (V, N) real COPEs
    varw = (eff[pick] / tval[pick]) ** 2               # (V, N) real VARCOPEs
    return beta.astype(np.float32), varw.astype(np.float32)


def real_flame_localizer(n_subjects: int = 40, n_vox: int = 8192, seed: int = 0
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Real two-level FLAME input from the Brainomics localizer: per-subject
    level-1 effects ``beta_subject (V, N)``, their **heterogeneous** within
    variances ``var_within (V, N) = (COPE / t)^2``, and the group-mean design
    ``X_group (N, 1)`` (a one-sample group test -- the canonical FLAME use).

    Cached as ``.npz`` so the ref workers read numpy, not nilearn.  Matches
    the ``flame_input`` contract (same shape) so the case branches cleanly.'''
    path = os.path.join(_CACHE, f'localizer_flame_{n_subjects}s_{n_vox}v.npz')
    if os.path.exists(path):
        z = np.load(path)
        beta, varw = z['beta'], z['varw']
    else:
        beta, varw = _materialise(n_subjects, n_vox, seed)
        np.savez(path, beta=beta, varw=varw)
    x_group = np.ones((beta.shape[1], 1), np.float32)
    return beta, varw, x_group


def _site_groups(n_subjects: int) -> np.ndarray:
    '''Integer acquisition-site label per subject (the reml random factor),
    in the same subject order as the maps.  nilearn imported here only.'''
    from nilearn import datasets as ds

    ev = ds.fetch_localizer_contrasts(
        [_CONTRAST], n_subjects=n_subjects, get_tmaps=True, data_dir=_CACHE
    )['ext_vars']
    sites = np.asarray(ev[_REML_GROUP].astype(str).values)
    return np.unique(sites, return_inverse=True)[1].astype(np.int64)


def real_reml_localizer(n_subjects: int = 40, n_vox: int = 2048, seed: int = 0
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]:
    '''Real one-way random-intercept input from the localizer: per-subject
    level-1 effects as ``Y (V, N)`` grouped by acquisition **site** (the
    canonical neuroimaging random-effect factor) -> the intercept ``X (N, 1)``,
    site indicators ``Z (N, k)``, and integer ``groups (N,)`` (statsmodels).

    ``real_full`` (no oracle): real + unbalanced, so the balanced closed form
    is inapplicable; correctness is **agreement with statsmodels MixedLM** on
    the real data.  NB the localizer has **k=2** sites -- a thin between-group
    df (a caveat); the cross-tool agreement benchmark is still valid.
    Matches the ``balanced_oneway`` return contract so the case branches
    cleanly.  Cached as ``.npz`` (ref workers read numpy, not nilearn).'''
    path = os.path.join(_CACHE, f'localizer_reml_{n_subjects}s_{n_vox}v.npz')
    if os.path.exists(path):
        z = np.load(path)
        y, groups = z['y'], z['groups']
    else:
        y, _varw = _materialise(n_subjects, n_vox, seed)  # COPEs as Y (V, N)
        groups = _site_groups(n_subjects)
        np.savez(path, y=y, groups=groups)
    k = int(groups.max()) + 1
    x = np.ones((y.shape[1], 1), np.float32)
    z_design = np.eye(k, dtype=np.float32)[groups]
    return y, x, z_design, groups
