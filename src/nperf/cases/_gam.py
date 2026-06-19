# -*- coding: utf-8 -*-
"""Shared helpers for the GAM/spline cases (``nitrix.stats.gam``).

The mass-univariate penalised-spline GAM: per voxel, fit ``y ~ s(x)`` (a
B-spline / P-spline smooth) with REML smoothing-parameter selection.  The fair
gold standard is **R ``mgcv::gam``** (P-spline ``bs='ps'``, ``method='REML'``),
looped one fit per voxel -- the batched(nitrix)-vs-looped(mgcv) story.  The
*fitted smooth* ``yhat`` is the convention-robust quantity to compare (the
basis parametrisation differs between tools, but the fitted curve does not --
VERIFIED nitrix vs mgcv agree to ~2e-6).  A numpy penalised-least-squares
oracle at a FIXED lambda validates the fitting core exactly (in the tests).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Tuple

import numpy as np


def gam_data(n_vox: int, n_obs: int, seed: int = 0, noise: float = 0.3
             ) -> Tuple[np.ndarray, np.ndarray]:
    '''One-covariate smooth-regression data: ``x`` sorted on [0, 1], and
    ``Y (V, N)`` = a smooth truth ``f(x) = sin(2 pi x)`` + per-voxel
    N(0, noise) noise.  Returns ``(x (N,), Y (V, N))``.'''
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n_obs)).astype(np.float32)
    f = np.sin(2.0 * np.pi * x)
    Y = (f[None, :] + noise * rng.standard_normal((n_vox, n_obs)))
    return x, Y.astype(np.float32)


def np_penalised_ls(D: np.ndarray, S_full: np.ndarray, lam: float
                    ) -> Callable[..., Any]:
    '''Fixed-lambda penalised-least-squares oracle (fp64): the GAM fit reduces
    to ``beta = (D'D + lam S)^-1 D'y`` at a pinned ``lam``; returns the fitted
    ``yhat (V, N)``.  Validates nitrix's penalised-IRLS core (the tests pin
    nitrix at the same lam via ``lam_floor=lam_ceil``).'''
    Dd = np.asarray(D, np.float64)
    A = Dd.T @ Dd + lam * np.asarray(S_full, np.float64)

    def run(Y: Any) -> Any:
        Yd = np.asarray(Y, np.float64)
        beta = np.linalg.solve(A, Dd.T @ Yd.T)   # (cols, V)
        return (Dd @ beta).T                       # (V, N)
    return run


# One Rscript loops mgcv::gam over the V voxels (R startup paid once).  The
# P-spline ``bs='ps'`` + ``method='REML'`` matches nitrix's bspline_basis
# (difference penalty) + REML-style selection; writes the fitted yhat (V, N).
def _r_mgcv_code(n_basis: int, fit: bool) -> str:
    body = (
        'm<-gam(Y[i,]~s(x,k=K,bs="ps"),method="REML");'
        'out[i,]<-as.numeric(fitted(m))' if fit else 'out[i,]<-Y[i,]*0')
    return (
        'a<-commandArgs(TRUE);d<-a[1];suppressMessages(library(mgcv))\n'
        'Y<-as.matrix(read.csv(file.path(d,"Y.csv"),header=FALSE))\n'
        'x<-scan(file.path(d,"x.csv"),quiet=TRUE);K<-%d\n'
        'V<-nrow(Y);out<-matrix(0,V,length(x))\n'
        'for(i in 1:V){%s}\n'
        'write.table(out,file.path(d,"out.csv"),sep=",",'
        'row.names=FALSE,col.names=FALSE)\n' % (n_basis, body))


def _run_r_gam(code: str, Y: Any, x: Any) -> np.ndarray:
    import subprocess
    import tempfile

    rscript = os.environ.get('NPERF_RSCRIPT',
                             '/scratch/nperf/renv/bin/Rscript')
    yh = np.asarray(Y, np.float64)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        np.savetxt(f'{d}/Y.csv', yh, delimiter=',')
        np.savetxt(f'{d}/x.csv', np.asarray(x, np.float64))
        with open(f'{d}/g.R', 'w') as f:
            f.write(code)
        subprocess.run([rscript, f'{d}/g.R', d], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = np.loadtxt(f'{d}/out.csv', delimiter=',')
    return out.reshape(yh.shape)


def r_mgcv_gam(n_basis: int) -> Callable[..., Any]:
    '''R **mgcv::gam** (P-spline, REML) looped per voxel -> fitted ``yhat
    (V, N)``.  THE GAM gold standard; CPU-only, looped, file-coupled -> a
    slow baseline.'''
    code = _r_mgcv_code(n_basis, fit=True)
    return lambda Y, x: _run_r_gam(code, Y, x)


def r_mgcv_iofloor(n_basis: int) -> Callable[..., Any]:
    '''I/O floor for ``r_mgcv_gam``: the same CSV write + R startup (+ mgcv
    load) + read, but NO fit -- the file-coupling artifact nitrix never pays
    (economic subtracts the same-namespace floor). Returns zeros.'''
    code = _r_mgcv_code(n_basis, fit=False)
    return lambda Y, x: _run_r_gam(code, Y, x) * 0.0


# -- spline-type breadth (cyclic / thin-plate / tensor-product) --------------
def gam_data_2d(n_vox: int, n_obs: int, seed: int = 0, noise: float = 0.2
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Two-covariate interaction-smooth data: ``x1, x2`` on [0, 1] and
    ``Y (V, N)`` = ``f(x1, x2) = sin(2 pi x1) cos(2 pi x2)`` + noise (for the
    tensor-product ``te(x1, x2)`` smooth).  Returns ``(x1, x2, Y)``.'''
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, n_obs).astype(np.float32)
    x2 = rng.uniform(0.0, 1.0, n_obs).astype(np.float32)
    f = np.sin(2.0 * np.pi * x1) * np.cos(2.0 * np.pi * x2)
    Y = (f[None, :] + noise * rng.standard_normal((n_vox, n_obs)))
    return x1, x2, Y.astype(np.float32)


def _r_smooth_code(term: str, fit: bool) -> str:
    '''mgcv loop for an arbitrary smooth ``term`` (e.g. ``s(x1,bs="cc",k=12)``
    or ``te(x1,x2,k=c(6,6))``); covariates are the columns of ``X.csv`` bound
    as ``x1, x2, ...``.'''
    body = (('m<-gam(Y[i,]~%s,method="REML");out[i,]<-as.numeric(fitted(m))'
             % term) if fit else 'out[i,]<-Y[i,]*0')
    return (
        'a<-commandArgs(TRUE);d<-a[1];suppressMessages(library(mgcv))\n'
        'Y<-as.matrix(read.csv(file.path(d,"Y.csv"),header=FALSE))\n'
        'X<-as.matrix(read.csv(file.path(d,"X.csv"),header=FALSE))\n'
        'for(j in 1:ncol(X)) assign(paste0("x",j), X[,j])\n'
        'V<-nrow(Y);out<-matrix(0,V,nrow(X))\n'
        'for(i in 1:V){%s}\n'
        'write.table(out,file.path(d,"out.csv"),sep=",",'
        'row.names=FALSE,col.names=FALSE)\n' % body)


def _run_r_smooth(code: str, Y: Any, X: Any) -> np.ndarray:
    import subprocess
    import tempfile

    rscript = os.environ.get('NPERF_RSCRIPT',
                             '/scratch/nperf/renv/bin/Rscript')
    yh = np.asarray(Y, np.float64)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        np.savetxt(f'{d}/Y.csv', yh, delimiter=',')
        np.savetxt(f'{d}/X.csv', np.asarray(X, np.float64), delimiter=',')
        with open(f'{d}/g.R', 'w') as f:
            f.write(code)
        subprocess.run([rscript, f'{d}/g.R', d], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = np.loadtxt(f'{d}/out.csv', delimiter=',')
    return out.reshape(yh.shape)


def r_mgcv_smooth(term: str) -> Callable[..., Any]:
    '''R mgcv::gam with the given smooth ``term``, looped per voxel -> fitted
    ``yhat (V, N)`` (the gold standard for the spline-type breadth).'''
    code = _r_smooth_code(term, fit=True)
    return lambda Y, X: _run_r_smooth(code, Y, X)


def r_mgcv_smooth_iofloor(term: str) -> Callable[..., Any]:
    '''I/O floor for ``r_mgcv_smooth`` (CSV + R/mgcv startup, no fit).'''
    code = _r_smooth_code(term, fit=False)
    return lambda Y, X: _run_r_smooth(code, Y, X) * 0.0


def build_smooth_case(param: Any) -> Any:
    '''Shared builder for the spline-type breadth cases (cyclic / thinplate /
    tensor): fit a GAM with the matching nitrix basis, compare the fitted
    smooth ``yhat`` to mgcv's matching smooth (cc / tp / te).  ``fp64_reference
    =None`` -- correctness is mgcv agreement (validated out-of-band).'''
    import jax
    import jax.numpy as jnp
    from nitrix.stats.basis import (
        bspline_basis,
        cyclic_cubic_basis,
        tensor_product_basis,
        thinplate_regression_basis,
    )
    from nitrix.stats.gam import gam_fit

    from ._base import ApproxBaseline, BuiltPoint, SlowBaseline  # noqa: F401
    smooth = param['smooth']
    v, big_n, k = int(param['V']), int(param['N']), int(param['n_basis'])
    seed = param.get('seed', 0)
    if smooth == 'tensor':
        x1, x2, Y = gam_data_2d(v, big_n, seed)
        sb = tensor_product_basis([bspline_basis(jnp.asarray(x1), k),
                                   bspline_basis(jnp.asarray(x2), k)])
        big_x = np.stack([x1, x2], axis=1)
        term = 'te(x1,x2,k=c(%d,%d))' % (k, k)
    else:
        x, Y = gam_data(v, big_n, seed)
        if smooth == 'cyclic':
            sb = cyclic_cubic_basis(jnp.asarray(x), k)
            term = 's(x1,bs="cc",k=%d)' % k
        else:  # thinplate
            sb = thinplate_regression_basis(jnp.asarray(x), k)
            term = 's(x1,bs="tp",k=%d)' % k
        big_x = x[:, None]
    big_d = jax.block_until_ready(jnp.concatenate(
        [jnp.ones((big_n, 1), sb.design.dtype), sb.design], axis=1))
    jY = jax.block_until_ready(jnp.asarray(Y))

    def _nitrix(y: Any) -> Any:
        return (big_d @ gam_fit(y, [sb]).coef.T).T

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (jY,) if framework == 'jax' else (Y, big_x)

    return BuiltPoint(
        baselines={
            'nitrix-jax': ('jax', _nitrix),
            'R.mgcv': ('r', r_mgcv_smooth(term)),
            'R.iofloor': ('r', r_mgcv_smooth_iofloor(term)),
        },
        inputs_for=inputs_for, fp64_reference=None,
        ratio_reference='nitrix-jax',
        fidelity_note=(
            'no exact cross-tool oracle; the fitted smooth yhat agrees with '
            'mgcv (matching %s smooth) -- VERIFIED corr ~1.0 '
            '(cyclic/thinplate) / ~0.99 (tensor), validated out-of-band.'
            % smooth),
    )
