# -*- coding: utf-8 -*-
"""Shared helpers for the voxelwise LME cases (``nitrix.stats.lme``).

The benchmark uses a **balanced one-way random-intercept** design (``k`` groups
x ``n`` per group, ``N = k·n`` subjects, shared intercept design ``X`` and
group-indicator random-effect design ``Z``), because for that design the REML
variance components have a **closed form** -- a reliable, vectorised fp64
oracle that needs no iterative solver and no external library.

Verified (2026-06-02): nitrix ``reml_fit`` matches this closed form *exactly*
on the variance components and the fixed effect, while ``statsmodels.MixedLM``
-- the canonical CPU library -- can fail to converge near the
variance-component boundary (small ``sigma_b^2``).  So the **closed form is the
oracle** (truth); ``statsmodels`` is the canonical-but-flaky *baseline* (the
real-world looped-CPU comparison), and the few boundary divergences are a
finding, not a bug in nitrix.

Output convention: every estimator returns ``(V, 3)`` columns
``[beta (intercept), sigma_b^2, sigma_e^2]`` so the fidelity compare and the
ratio are over the same quantity.
"""
from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np


def balanced_oneway(
    n_vox: int, k: int, n: int, seed: int = 0,
    sigma_b_sq: float = 1.0, sigma_e_sq: float = 1.0, grand: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Balanced one-way random-intercept data.

    Returns ``(Y (V, N), X (N, 1), Z (N, k), groups (N,))`` -- ``X`` is the
    shared intercept, ``Z`` the group indicators, ``groups`` the integer group
    label per subject (for statsmodels).  ``sigma_b_sq`` is kept comfortably
    away from 0 so the canonical statsmodels baseline mostly converges.'''
    rng = np.random.default_rng(seed)
    big_n = k * n
    groups = np.repeat(np.arange(k), n).astype(np.int64)
    X = np.ones((big_n, 1), np.float32)
    Z = np.eye(k, dtype=np.float32)[groups]
    b = rng.standard_normal((n_vox, k)) * np.sqrt(sigma_b_sq)
    eps = rng.standard_normal((n_vox, big_n)) * np.sqrt(sigma_e_sq)
    Y = (grand + b[:, groups] + eps).astype(np.float32)
    return Y, X, Z, groups


def closed_form_reml(Y: np.ndarray, k: int, n: int) -> np.ndarray:
    '''Closed-form balanced one-way REML -> ``(V, 3)`` fp64
    ``[beta, sigma_b^2, sigma_e^2]``.

    For a balanced one-way random-effects model the REML estimates equal the
    ANOVA estimates: ``sigma_e^2 = MSW``, ``sigma_b^2 = max((MSB-MSW)/n, 0)``,
    ``beta = grand mean`` (exact, no iteration).'''
    n_vox, big_n = Y.shape
    yg = Y.reshape(n_vox, k, n)
    group_mean = yg.mean(2)                       # (V, k)
    grand_mean = Y.mean(1)                         # (V,)
    ms_between = n * ((group_mean - grand_mean[:, None]) ** 2).sum(1) / (k - 1)
    ms_within = ((yg - group_mean[:, :, None]) ** 2).sum((1, 2)) / (big_n - k)
    sigma_e_sq = ms_within
    sigma_b_sq = np.maximum((ms_between - ms_within) / n, 0.0)
    return np.stack([grand_mean, sigma_b_sq, sigma_e_sq], axis=-1)


def flame_input(
    n_vox: int, big_n: int, seed: int = 0,
    sigma_b_sq: float = 1.0, s2: float = 0.3, grand: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''FLAME two-level data: per-voxel, per-subject level-1 effects
    ``beta_subject (V, N)`` with a **constant known** within-variance ``s2``
    (so the single-parameter REML has a closed form -- see
    ``flame_closed_form``), and the shared intercept group design.

    Returns ``(beta_subject (V, N), var_within (V, N), X_group (N, 1))``.'''
    rng = np.random.default_rng(seed)
    x_group = np.ones((big_n, 1), np.float32)
    b = rng.standard_normal((n_vox, big_n)) * np.sqrt(sigma_b_sq)
    e = rng.standard_normal((n_vox, big_n)) * np.sqrt(s2)
    beta_subject = (grand + b + e).astype(np.float32)
    var_within = np.full((n_vox, big_n), s2, np.float32)
    return beta_subject, var_within, x_group


def flame_closed_form(
    beta_subject: np.ndarray, x_group: np.ndarray, s2: float,
) -> np.ndarray:
    '''Closed-form FLAME REML for **constant** within-variance ``s2`` -> ``(V,
    2)`` fp64 ``[gamma, sigma_b^2]``.

    With ``s_i^2 = s2`` the model covariance is ``(sigma_b^2 + s2) I``, so the
    REML reduces to GLS == OLS for ``gamma`` and the residual variance for the
    total: ``sigma_b^2 = max(||resid||^2/(N-p) - s2, 0)`` (exact, no
    iteration).'''
    big_n, p = x_group.shape
    xtx_inv = np.linalg.inv(x_group.T @ x_group)        # (p, p)
    gamma = beta_subject @ x_group @ xtx_inv.T          # (V, p)
    resid = beta_subject - gamma @ x_group.T            # (V, N)
    tau2 = (resid ** 2).sum(1) / (big_n - p)
    sigma_b_sq = np.maximum(tau2 - s2, 0.0)
    return np.stack([gamma[:, 0], sigma_b_sq], axis=-1)


def statsmodels_reml(Y: Any, X: Any, groups: Any) -> np.ndarray:
    '''Looped ``statsmodels.MixedLM`` REML -> ``(V, 3)``
    ``[beta, sigma_b^2, sigma_e^2]``.  statsmodels imported lazily (only this
    baseline's worker needs it -- the base env); convergence warnings are
    silenced (boundary non-convergence is expected, surfaced via fidelity).'''
    import warnings

    import statsmodels.api as sm

    yh = np.asarray(Y, np.float64)
    xh = np.asarray(X, np.float64)
    out = np.empty((yh.shape[0], 3), np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for v in range(yh.shape[0]):
            m = sm.MixedLM(yh[v], xh, groups=groups).fit(reml=True)
            out[v] = (m.fe_params[0],
                      float(np.asarray(m.cov_re)[0, 0]), m.scale)
    return out


# R lme4 ``lmer`` is THE gold-standard mixed-effects REML; for the balanced
# one-way design ``y ~ 1 + (1|group)`` its REML estimate equals the closed-form
# oracle (and what nitrix reml_fit targets).  One Rscript loops lmer over the V
# voxels (so we pay R startup once, not per voxel) and writes (V, 3).
_R_LME4_CODE = r'''
args <- commandArgs(trailingOnly=TRUE); d <- args[1]
suppressMessages(library(lme4))
Y <- as.matrix(read.csv(file.path(d, "Y.csv"), header=FALSE))   # V x N
g <- factor(scan(file.path(d, "grp.csv"), quiet=TRUE))          # length N
V <- nrow(Y); out <- matrix(0.0, V, 3)
ctrl <- lmerControl(check.conv.singular="ignore",
                    check.nobs.vs.nlev="ignore", check.nobs.vs.nRE="ignore")
for (i in 1:V) {
  y <- as.numeric(Y[i, ])
  m <- suppressMessages(suppressWarnings(
        lmer(y ~ 1 + (1 | g), REML=TRUE, control=ctrl)))
  vc <- as.data.frame(VarCorr(m))
  sb <- vc$vcov[vc$grp == "g" & is.na(vc$var2)]
  se <- vc$vcov[vc$grp == "Residual"]
  out[i, ] <- c(as.numeric(fixef(m))[1], sb, se)
}
write.table(out, file.path(d, "out.csv"), sep=",",
            row.names=FALSE, col.names=FALSE)
'''

# I/O floor: boot Rscript, read the SAME Y/grp CSVs, write a trivial (V, 3) --
# NO library(lme4), NO fit.  Isolates the file-coupling artifact R pays and
# nitrix never does: the CSV write/read + the R interpreter startup (which is
# a LARGE fraction of wall-clock at small V).  economic_report subtracts the
# same-namespace ``r.iofloor`` (``compute = R.lme4 - floor``); mirrors
# ``flameo_iofloor`` (which runs the cheap fslmaths, not flameo).
_R_IOFLOOR_CODE = r'''
args <- commandArgs(trailingOnly=TRUE); d <- args[1]
Y <- as.matrix(read.csv(file.path(d, "Y.csv"), header=FALSE))
g <- scan(file.path(d, "grp.csv"), quiet=TRUE)
out <- matrix(0.0, nrow(Y), 3)   # no fit -- IO + R startup only
write.table(out, file.path(d, "out.csv"), sep=",",
            row.names=FALSE, col.names=FALSE)
'''


def _r_run(code: str, Y: Any, groups: Any) -> np.ndarray:
    '''Write ``Y``/``groups`` to a temp dir, run ``code`` via Rscript, read the
    ``(V, 3)`` ``out.csv`` back.  Rscript at ``$NPERF_RSCRIPT`` (default
    ``/scratch/nperf/renv/bin/Rscript``).'''
    import os
    import subprocess
    import tempfile

    rscript = os.environ.get('NPERF_RSCRIPT',
                             '/scratch/nperf/renv/bin/Rscript')
    yh = np.asarray(Y, np.float64)
    grp = np.asarray(groups).ravel().astype(np.int64)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as d:
        np.savetxt(f'{d}/Y.csv', yh, delimiter=',')
        np.savetxt(f'{d}/grp.csv', grp, fmt='%d')
        with open(f'{d}/fit.R', 'w') as f:
            f.write(code)
        subprocess.run([rscript, f'{d}/fit.R', d], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = np.loadtxt(f'{d}/out.csv', delimiter=',')
    return out.reshape(yh.shape[0], 3)


def r_lme4_reml(Y: Any, groups: Any) -> np.ndarray:
    '''Looped R **lme4 ``lmer``** REML (the gold-standard mixed-model fit) ->
    ``(V, 3)`` ``[beta, sigma_b^2, sigma_e^2]``.  Runs ONE Rscript that loops
    ``lmer(y ~ 1 + (1|g), REML=TRUE)`` over the V voxels.  CPU-only, looped,
    and file-coupled -> a slow baseline whose CSV+startup I/O is timed by
    ``r_lme4_iofloor`` (economic subtracts it).'''
    return _r_run(_R_LME4_CODE, Y, groups)


def r_lme4_iofloor(Y: Any, groups: Any) -> np.ndarray:
    '''I/O floor for ``r_lme4_reml``: the same CSV write + Rscript startup +
    read-back, but NO lme4 / no fit -- the file-coupling + R-boot artifact
    nitrix never pays (economic_report subtracts the same-namespace floor).
    Returns zeros (timing only, not scored).'''
    return _r_run(_R_IOFLOOR_CODE, Y, groups) * 0.0


# -- FLAME two-level external baselines (flame_two_level case) ---------------
# The fair competitor for nitrix's known-within-variance two-level REML is the
# upstream tool FSL **FLAME** (`flameo`), which the case docstring long flagged
# as "to be revisited in the external-tool workstream".  It is file-coupled
# (NIfTI + VEST design), like the AFNI/FSL registration refs, so the wrapper
# pays a NIfTI round-trip the in-memory nitrix op never does -- subtracted via
# the `flameo_iofloor` no-op (economic_report).  Spin-up: tools/
# setup_neuro_refs.sh (/scratch is ephemeral); see README.  Both flameo and the
# statsmodels meta-analysis LOOP one fit per voxel, so they are slow_baselines
# whose batched-vs-looped gap GROWS with the voxel batch V.


def _flame_box(v: int) -> Tuple[int, int, int]:
    '''A cube-ish `(nx, ny, nz)` with `nx*ny*nz >= v` and every dim < 32767
    (the NIfTI-1 int16 `dim[]` limit): V voxels are laid row-major into the
    first `v` cells (the rest masked out).  ~cube keeps each axis tiny
    (V=262144 -> 64^3), so no axis ever approaches the limit.'''
    import math

    a = max(1, int(round(v ** (1.0 / 3.0))))
    while a * a * math.ceil(v / (a * a)) < v:  # guard the rounding
        a += 1
    return a, a, int(math.ceil(v / (a * a)))


def _write_vest(path: str, mat: np.ndarray, kind: str) -> None:
    '''Write an FSL VEST design file (`design.mat` / `.con` / `.grp`).  `kind`
    == 'con' uses a `/NumContrasts` header; otherwise `/NumPoints`.'''
    mat = np.atleast_2d(np.asarray(mat, np.float64))
    nrow, ncol = mat.shape
    with open(path, 'w') as f:
        head = 'NumContrasts' if kind == 'con' else 'NumPoints'
        f.write(f'/NumWaves {ncol}\n/{head} {nrow}\n/Matrix\n')
        for row in mat:
            f.write(' '.join(f'{x:.8f}' for x in row) + '\n')


def _flame_write_inputs(d: str, beta: np.ndarray, varw: np.ndarray,
                        x_group: np.ndarray) -> None:
    '''Write the flameo inputs into dir `d`: cope/varcope (V voxels laid into a
    cube-ish (nx,ny,nz) box -- see `_flame_box`; NIfTI-1 `dim[]` is int16, so a
    flat (V,1,1) column overflows at V>32767), N subjects in the 4th dim, with
    a mask selecting the first V cells; and the VEST design (group matrix),
    contrast (the 1st EV -> matches `gamma_hat[:,0]`), and covariance-split
    (single variance group).'''
    import nibabel as nib

    beta = np.asarray(beta, np.float32)
    varw = np.asarray(varw, np.float32)
    v, big_n = beta.shape
    p = x_group.shape[1]
    nx, ny, nz = _flame_box(v)
    tot = nx * ny * nz
    eye = np.eye(4)

    def _box4(flat: np.ndarray) -> np.ndarray:  # (V,N) -> (nx,ny,nz,N), padded
        full = np.zeros((tot, big_n), np.float32)
        full[:v] = flat
        return full.reshape(nx, ny, nz, big_n)

    mask = np.zeros(tot, np.float32)
    mask[:v] = 1.0
    nib.save(nib.Nifti1Image(_box4(beta), eye), f'{d}/cope.nii.gz')
    nib.save(nib.Nifti1Image(_box4(varw), eye), f'{d}/varcope.nii.gz')
    nib.save(nib.Nifti1Image(mask.reshape(nx, ny, nz), eye),
             f'{d}/mask.nii.gz')
    _write_vest(f'{d}/design.mat', np.asarray(x_group, np.float64), 'mat')
    con = np.zeros((1, p))
    con[0, 0] = 1.0                                  # the 1st EV (gamma)
    _write_vest(f'{d}/design.con', con, 'con')
    _write_vest(f'{d}/design.grp', np.ones((big_n, 1)), 'grp')  # one var group


def flameo_flame1() -> Callable[..., Any]:
    '''FSL **FLAME** (`flameo --runmode=flame1`) -- THE upstream tool for the
    two-level mixed-effects group model, and the fair competitor for
    `flame_two_level`.  flame1 is the fast mixed-effects estimate (the stage-1
    EM that nitrix's single-parameter REML matches; flame12 adds a slow stage-2
    MCMC, not benched).  flameo iterates **voxel-by-voxel** on CPU -> the
    batched-vs-looped story.  Reads `stats/pe1` (the group effect gamma) and
    `stats/mean_random_effects_var1` (sigma_b^2); returns `(V, 2)` =
    `[gamma, sigma_b^2]` to match nitrix.  Binary at `$NPERF_FSL_DIR/bin`
    (default `/scratch/nperf/fsl`); not jit (full wall-clock).'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(beta: Any, varw: Any, x_group: Any) -> Any:
        import subprocess
        import tempfile

        import nibabel as nib

        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        v = np.asarray(beta).shape[0]
        with tempfile.TemporaryDirectory(
                dir=os.environ.get('TMPDIR')) as d:
            _flame_write_inputs(d, beta, varw, x_group)
            subprocess.run(
                [os.path.join(fsldir, 'bin', 'flameo'),
                 '--cope=cope.nii.gz', '--varcope=varcope.nii.gz',
                 '--mask=mask.nii.gz', '--ld=stats', '--dm=design.mat',
                 '--tc=design.con', '--cs=design.grp', '--runmode=flame1'],
                cwd=d, check=True, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            s = f'{d}/stats'
            # the box is masked to the first v cells -> flatten row-major +
            # slice (matches how _flame_write_inputs laid the V voxels in).
            gamma = np.asarray(
                nib.load(f'{s}/pe1.nii.gz').get_fdata(),
                np.float64).reshape(-1)[:v]
            sigb2 = np.asarray(
                nib.load(f'{s}/mean_random_effects_var1.nii.gz').get_fdata(),
                np.float64).reshape(-1)[:v]
        return np.stack([gamma, sigb2], -1)

    return run


def flameo_iofloor() -> Callable[..., Any]:
    '''I/O floor for `flameo_flame1`: the same NIfTI writes (cope + varcope) +
    a subprocess (`fslmaths -mul 1`, read 4D + write 4D) + read-back, but NO
    FLAME fit -- so its wall-clock is the file-coupling artifact nitrix never
    pays.  economic_report subtracts the same-namespace `fsl.iofloor`
    (`compute = flameo - floor`).  Approximate (flameo also writes ~13 small
    output volumes), but captures the dominant V*N input write + subprocess +
    read.'''
    import os

    fsldir = os.environ.get('NPERF_FSL_DIR', '/scratch/nperf/fsl')

    def run(beta: Any, varw: Any, x_group: Any) -> Any:
        import subprocess
        import tempfile

        import nibabel as nib

        env = {**os.environ, 'FSLDIR': fsldir, 'FSLOUTPUTTYPE': 'NIFTI_GZ'}
        v = np.asarray(beta).shape[0]
        with tempfile.TemporaryDirectory(
                dir=os.environ.get('TMPDIR')) as d:
            _flame_write_inputs(d, beta, varw, x_group)  # cope+varcope+design
            subprocess.run(
                [os.path.join(fsldir, 'bin', 'fslmaths'), 'cope.nii.gz',
                 '-mul', '1', 'floorout.nii.gz'],
                cwd=d, check=True, env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _ = np.asarray(nib.load(f'{d}/floorout.nii.gz').get_fdata())
        return np.zeros((v, 2), np.float64)  # floor: timing only, not scored

    return run


def statsmodels_flame() -> Callable[..., Any]:
    '''Looped `statsmodels.stats.meta_analysis.combine_effects` -- the
    known-within-variance two-level model IS a random-effects meta-analysis
    (combine N per-subject effects with known variances -> pooled effect +
    between-subject heterogeneity tau^2 = sigma_b^2), fit **one voxel at a
    time** (the looped-CPU competitor).  NOTE: combine_effects' tau^2 is
    Paule-Mandel (`method_re='iterated'`), NOT REML, so it **diverges** from
    the REML oracle at the variance-component boundary (tau^2 -> 0) -- a
    documented finding (cf. statsmodels MixedLM in `reml_fit`), surfaced via
    fidelity, not a bug.  Intercept design only (p=1: the pooled mean).'''
    def run(beta: Any, varw: Any, x_group: Any) -> Any:
        import warnings

        from statsmodels.stats.meta_analysis import combine_effects

        b = np.asarray(beta, np.float64)
        w = np.asarray(varw, np.float64)
        out = np.empty((b.shape[0], 2), np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(b.shape[0]):
                res = combine_effects(b[i], w[i], method_re='iterated')
                out[i] = (res.mean_effect_re, res.tau2)
        return out

    return run
