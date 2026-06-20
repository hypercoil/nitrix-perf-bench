# -*- coding: utf-8 -*-
"""Shared helpers for the voxelwise GLMM case (``nitrix.stats.glmm.glmm_fit``).

Mass-univariate generalised linear *mixed* models by PQL: per element
(voxel), ``g(E[y|b]) = X beta + b[group]`` under a GLM ``family``, with an
optional random **slope** (``z``).  The benchmark sweeps a dense range of
small, fast scales per modelling PATH -- ``family`` x ``structure`` (intercept
/ unstructured slope) x ``method`` (pql / agq) x level-count tier (few
``q<=64`` dense vs many ``q>64`` structured) -- so brain-scale cost is
*extrapolated* (see ``tools/extrapolate_report.py``), keeping the
numerically-hard cells (the robust unstructured-slope solver, AGQ for binary
clusters) measured too.

Data is a balanced one-way design with a planted fixed effect + random
effect(s), so the FIXED effect recovers the planted truth (a deterministic
recovery check) even where the variance component is PQL-attenuated.  The
Gaussian random-intercept GLMM *is* the LME, so its oracle is the closed-form
balanced REML (``_lme.closed_form_reml``); the non-Gaussian paths have no
closed form (fidelity = agreement with the R reference, when present, +
finiteness).

The looped-CPU gold standard is **R ``mgcv::gam(family=, s(g, bs='re'),
method='REML')``** -- the §1.2 estimator nitrix's PQL matches -- run one fit
per voxel in a single Rscript (so R startup is paid once).  It is a slow
baseline, infeasible at brain scale: exactly the curve the extrapolation tool
fits at small scale and projects forward.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np


def glmm_data(
    family: str, n_vox: int, q: int, n_per: int, structure: str = 'intercept',
    seed: int = 0, sigma_b: float = 0.7, beta0: float = 0.3,
    beta1: float = 0.5, sigma_e: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray],
           Dict[str, float]]:
    '''Balanced GLMM data with a planted fixed + random effect.

    Returns ``(Y (V, N), X (N, p), group (N,), z (N, r) | None, truth)`` where
    ``truth`` carries the planted ``beta0`` / ``beta1`` / ``sigma_b_sq`` (for
    recovery checks).  ``structure='intercept'`` -> ``X=[1]``, ``z=None`` (the
    scalar ``(1|g)``); ``'unstructured'`` / ``'diagonal'`` -> a random
    **slope**, ``X=z=[1, x]`` with continuous ``x`` (correlated ``(1+x|g)``
    / uncorrelated ``(x||g)``).  ``family`` shapes the response: ``gaussian``
    (identity + noise), ``binomial`` (Bernoulli 0/1, logit), ``poisson``
    (counts, log) -- low ``n_per`` keeps clusters small (the PQL-attenuation /
    challenging regime).'''
    rng = np.random.default_rng(seed)
    big_n = q * n_per
    group = np.repeat(np.arange(q), n_per).astype(np.int64)
    if structure == 'intercept':
        X = np.ones((big_n, 1), np.float32)
        z: Optional[np.ndarray] = None
        b = rng.standard_normal((n_vox, q)) * sigma_b
        eta = beta0 + b[:, group]                                  # (V, N)
    else:
        x = rng.standard_normal(big_n).astype(np.float32)
        X = np.stack([np.ones(big_n, np.float32), x], axis=1)      # (N, 2)
        z = X.copy()                                               # [1, x]
        b0 = rng.standard_normal((n_vox, q)) * sigma_b
        b1 = rng.standard_normal((n_vox, q)) * sigma_b
        eta = beta0 + beta1 * x + b0[:, group] + b1[:, group] * x
    if family == 'gaussian':
        Y = (eta + rng.standard_normal(eta.shape) * sigma_e)
    elif family == 'binomial':
        p = 1.0 / (1.0 + np.exp(-eta))
        Y = (rng.random(eta.shape) < p).astype(np.float64)
    elif family == 'poisson':
        Y = rng.poisson(np.exp(np.clip(eta, None, 5.0))).astype(np.float64)
    else:
        raise ValueError(f'unknown family {family!r}')
    truth = {'beta0': beta0, 'beta1': beta1, 'sigma_b_sq': sigma_b ** 2}
    return Y.astype(np.float32), X, group, z, truth


# One Rscript loops mgcv::gam(family=, s(g, bs="re"), method="REML") -- the PQL
# random-intercept estimator nitrix matches -- over V voxels (R startup paid
# once), writes (V, 2) = [intercept, sigma_b^2]. For a single bs="re" smooth
# mgcv's i.i.d. random-effect variance is scale/sp (scale=1 for binomial /
# poisson), a robust extraction that avoids parsing gam.vcomp.
_R_MGCV_CODE = r'''
args <- commandArgs(trailingOnly=TRUE); d <- args[1]; fam <- args[2]
suppressMessages(library(mgcv))
Y <- as.matrix(read.csv(file.path(d, "Y.csv"), header=FALSE))   # V x N
g <- factor(scan(file.path(d, "grp.csv"), quiet=TRUE))          # length N
famobj <- switch(fam, gaussian=gaussian(), binomial=binomial(),
                 poisson=poisson())
V <- nrow(Y); out <- matrix(0.0, V, 2)
for (i in 1:V) {
  y <- as.numeric(Y[i, ])
  m <- suppressMessages(suppressWarnings(
        gam(y ~ s(g, bs="re"), family=famobj, method="REML")))
  out[i, ] <- c(as.numeric(coef(m))[1], m$scale / m$sp[1])
}
write.table(out, file.path(d, "out.csv"), sep=",",
            row.names=FALSE, col.names=FALSE)
'''

# I/O floor: boot Rscript, read the SAME Y/grp CSVs, write a trivial (V, 2) --
# NO library(mgcv), NO fit.  Isolates the CSV write/read + R startup artifact R
# pays and nitrix never does (economic subtracts the same-namespace floor).
_R_IOFLOOR_CODE = r'''
args <- commandArgs(trailingOnly=TRUE); d <- args[1]
Y <- as.matrix(read.csv(file.path(d, "Y.csv"), header=FALSE))
g <- scan(file.path(d, "grp.csv"), quiet=TRUE)
out <- matrix(0.0, nrow(Y), 2)   # no fit -- IO + R startup only
write.table(out, file.path(d, "out.csv"), sep=",",
            row.names=FALSE, col.names=FALSE)
'''


def _r_run(code: str, Y: Any, group: Any, family: str) -> np.ndarray:
    '''Write ``Y``/``group`` to a temp dir, run ``code`` via Rscript (passing
    ``family``), read the ``(V, 2)`` ``out.csv`` back.  Rscript at
    ``$NPERF_RSCRIPT`` (default ``/scratch/nperf/renv/bin/Rscript``).'''
    rscript = os.environ.get('NPERF_RSCRIPT',
                             '/scratch/nperf/renv/bin/Rscript')
    yh = np.asarray(Y, np.float64)
    grp = np.asarray(group).ravel().astype(np.int64)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        np.savetxt(f'{d}/Y.csv', yh, delimiter=',')
        np.savetxt(f'{d}/grp.csv', grp, fmt='%d')
        with open(f'{d}/fit.R', 'w') as f:
            f.write(code)
        subprocess.run([rscript, f'{d}/fit.R', d, family], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = np.loadtxt(f'{d}/out.csv', delimiter=',')
    return out.reshape(yh.shape[0], 2)


def r_mgcv_glmm(Y: Any, group: Any, family: str) -> np.ndarray:
    '''Looped R **mgcv::gam** random-intercept GLMM (§1.2 PQL gold standard)
    -> ``(V, 2)`` ``[intercept, sigma_b^2]``.  ONE Rscript loops
    ``gam(y ~ s(g, bs="re"), family=, method="REML")`` over the V voxels.
    CPU-only, looped, file-coupled -> a slow baseline whose CSV+startup I/O is
    timed by ``r_mgcv_iofloor`` (economic subtracts it).'''
    return _r_run(_R_MGCV_CODE, Y, group, family)


def r_mgcv_iofloor(Y: Any, group: Any) -> np.ndarray:
    '''I/O floor for ``r_mgcv_glmm``: the same CSV write + Rscript startup +
    read-back, no mgcv / no fit -- the file-coupling artifact nitrix never pays
    (economic_report subtracts the same-namespace floor).  Returns zeros.'''
    return _r_run(_R_IOFLOOR_CODE, Y, group, 'gaussian') * 0.0
