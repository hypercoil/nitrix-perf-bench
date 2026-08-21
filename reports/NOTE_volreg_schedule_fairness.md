# NOTE (for the perf agent): match the iteration schedule across tools for volreg

**Date:** 2026-06-21 · from the nitrix registration-perf work (`perf/matrix-registration`).

## The fairness issue

Motion realignment (`volreg`) is a **small-displacement** problem and a second-order
per-frame solver converges in a handful of steps. Measured (both a synthetic
planted-motion series **and** a real resting-state 4D fMRI, `nilearn
fetch_development_fmri`, 30 frames):

- nitrix `volreg` recovers **identical** per-frame transforms at a tight schedule
  `(4,2,1)` as at a loose `(60,30,15)` — max |Δparam| = **2.3e-4 rad/vox** on real
  motion, bit-identical at `(12,6,3)` — for ~**8×** less time (GPU 42.9→5.5 ms,
  CPU 25→3.3 s for T=20 @ MNI-2mm).
- So the *headline number depends entirely on the schedule.* Benchmarking
  nitrix-tight vs an ANTs/AFNI/FSL default (or vice-versa) is **not** apples-to-
  apples (the same trap as the SyN `40x20x10` vs ANTs-default `40x20x0`).

## Ask

When you bench `volreg` (and motion realignment generally) against the community
tools, **match the iteration schedule on both sides**:

- **nitrix**: `RegistrationSpec(iterations=(4,2,1))` (or whatever you pick) — and
  report it. The current `RegistrationSpec` default (30/level) over-iterates
  motion correction ~25× and is not a realistic volreg setting.
- **AFNI `3dvolreg`**: `-maxite` / convergence; set it to the matched budget.
- **FSL `mcflirt`**: there is no direct iteration knob, but pin its cost/levels
  comparably and note the difference.
- **ANTs motion correction** (`antsMotionCorr` / `ants.motion_correction`): set
  `iterations=` to the matched per-level budget (its default is loose).

Also keep the **I/O-floor subtraction** for the CLI tools (3dvolreg/mcflirt are
~40% NIfTI read+write), as you already do, and prefer **real 4D fMRI** over a
synthetic plant for the recovery/quality claim (the plant is clean/noiseless and
can flatter a tight schedule — though here the real-data result agreed).

(nitrix also gained an opt-in **batch-aggregate early-exit** for volreg —
`spec.convergence=Convergence(...)`, IC path — which adapts the count to the
motion; but a tight *fixed* schedule is faster still when the motion regime is
known, since the vmap'd while_loop runs to the slowest frame.)

## How the tools actually do it (checked the source/binaries here)

Confirms the premise — **both converge early on small motion; neither runs a
fixed large count**, so matching to a convergence/tight budget is the *faithful*
comparison, not a handicap:

- **AFNI `3dvolreg`** (`afni_src/src/3dvolreg.c`): *single-resolution iterated
  linearised least-squares* (Gauss-Newton-like; `mri_3dalign`). Converges when
  max movement `< x_thresh=0.01 vox` **and** max rotation `< rot_thresh=0.02°`,
  capped at **`maxite=23`** (`VL_maxite`; `-twopass` raises it to 66). Heptic
  interpolation by default. → It *early-exits*; 23 is just the ceiling.
- **FSL `mcflirt`** (`fsl/src/fsl-mcflirt/mcflirt.cc`): coarse-to-fine search,
  `current_scale=8.0` mm → 4 mm, **default 3 stages** (`-stages 3`; 4 adds a
  final sinc stage), `normcorr` cost, `dof=6`. Each stage runs
  `MISCMATHS::optimise` (Powell) to a **parameter tolerance** (`param_tol *
  new_tolerance`, `new_tolerance≈0.8`) with an iteration cap — tolerance-based,
  not a fixed count.

**Matched-comparison recipe:** nitrix's analogue of their convergence is the
opt-in `Convergence` early-exit (IC path) — or a tight fixed schedule sized to
the few effective iterations they actually take. The structures differ
(3dvolreg = single-res iterated-LSQ + threshold; mcflirt = 8→4 mm Powell-to-tol;
nitrix = pyramid inverse-compositional GN), so the only truly fair axis is
**iso-accuracy** (wall-clock to reach the same recovered motion / inter-frame
variance), with the CLI I/O floor subtracted.
