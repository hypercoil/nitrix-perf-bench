# nitrix-perf-bench — domain-tool reference roadmap

> **Status.** Roadmap (not yet a locked decision-of-record) for adding the
> **canonical neuroimaging / medical-imaging tools** as references. It refines
> [`COVERAGE_MANDATE.md`](COVERAGE_MANDATE.md) §2.3 (references) and §5
> (separation of concerns) and rides the [`DESIGN.md`](DESIGN.md) §7 refs-env /
> pixi escape hatch. It does **not** change any locked decision; it plans *which
> reference libraries to add and how*.

## 0. Where this fits the mandate

Mandate §2.3 defines two reference roles: the numpy / scipy / sklearn **CPU
floor** ("what you'd write without nitrix") and a **strong on-target GPU bar**
(CuPy / cuDNN / torch-CUDA — the real kernel-vs-kernel question). The domain
tools here are a **third role that upgrades the floor's credibility**: for a
neuroimaging numerics library, the honest floor is not a hand-rolled numpy
snippet but *the tool a practitioner actually runs* (nilearn, FSL, …). They are
mostly CPU / CLI, so they are **floor-class references** in the §2.2 taxonomy
(`floor_only`), not the strong on-target GPU bar — that axis stays CuPy / cuDNN
/ torch-CUDA and is unchanged. Two of them (the file-coupled binaries) also add
a genuinely new comparison *mode* — **end-to-end-with-I/O** — that the in-memory
floor cannot express.

Separation of concerns (§5) is unchanged: every tool lives in a perf-bench
**refs env** (uv group, or DESIGN §7 pixi/container for the binaries); `nitrix`
never gains a dependency on any of them.

## 1. The organizing axis: in-memory primitive vs file-coupled pipeline

The decisive question per tool is **"does it expose the primitive *in memory*,
or only as a file-coupled end-to-end pipeline?"** — that sets whether a *fair
kernel-vs-kernel* comparison is even possible.

- **Class A — in-memory Python primitive** (kernel-vs-kernel, fair; drops into
  existing infra exactly like the `sklearn` / `statsmodels` providers):
  **nilearn, SimpleITK, ANTsPy**.
- **Class B — file-coupled CLI binary** (no in-memory API → *end-to-end with an
  I/O floor* only): **AFNI, FSL, FreeSurfer, Connectome Workbench**.
- **Class C — language bridge** (MATLAB / Octave): **SPM**.

## 2. Per-tool assessment

| Tool | Lang / dist · license | Install | Primitive access | I/O coupling | nitrix overlap | Class · Tier |
|---|---|---|---|---|---|---|
| **nilearn** | Python (numpy/scipy/sklearn/nibabel) · BSD | pip | in-memory fns on arrays | none | connectome **partialcorr / precision / tangent / cov / corr**; `signal.clean` (detrend / filter / confound); smoothing; resampling | A · **Tier 1** |
| **SimpleITK** | C++ ITK + Py bindings · Apache-2 | pip | in-memory `sitk.Image` ↔ numpy | none | **erode / dilate**, **distance_transform**, **gaussian**, **median_filter**, **bilateral_gaussian**, **spatial_transform** (Resample), **N4 bias**, histogram-matching (Nyul-Udupa) | A · **Tier 1** |
| **ANTsPy** | C++ ITK + Py bindings · Apache-2 | pip wheel / build | in-memory `ANTsImage` (numpy-backed) | minimal (some temp files) | smoothing, **N4 bias**, registration, `apply_transforms` (spatial_transform) | A · Tier 2 |
| **dipy** | Python (numpy / scipy / cython) · BSD-3 | pip | in-memory `dipy.align` on arrays | none | **registration**: `AffineRegistration` (rigid / 12-DOF affine, MI), `SymmetricDiffeomorphicRegistration` (SyN; SSD/CC) ≈ **rigid_register / affine_register / diffeomorphic_demons** | A · **Tier 2 (shipped)** |
| **AFNI** | C, binary suite · free (NIH) | installer / container | CLI on NIfTI/BRIK | **mandatory disk I/O** | **3dLME / 3dMEMA** (reml / flame), `3dTproject` (detrend / filter), `3dTcorrelate`, `3dBlurToFWHM`, `3dDespike` | B · Tier 3 |
| **FSL** | C++, binary suite · non-commercial | installer / container | CLI on NIfTI (fslpy ~file-based) | **mandatory disk I/O** | **FLAME** (flame_two_level), `susan` (smoothing), `fslmaths`, flirt / fnirt, melodic | B · Tier 3 |
| **FreeSurfer** | C/C++, large suite · license | installer + license | CLI on FS surface/volume files | I/O + format conversion | surface smoothing / sphere geometry (narrow) | B · Tier 4 |
| **Connectome Workbench** (`wb_command`) | C++, HCP/WashU · open-source | installer / container | CLI on CIFTI / GIFTI / NIfTI | **mandatory disk I/O** + CIFTI/GIFTI formats | **dense connectome** (`-cifti-correlation` ≈ cov/corr on grayordinates), surface/metric **smoothing**, **geodesic distance**, metric **resample**, parcellation — overlaps sphere / mesh-geometry + connectome | B · Tier 4 (boutique) |
| **SPM** | MATLAB toolbox · GPL | MATLAB (license) / Octave (partial) | MATLAB fns via `matlab.engine` / `oct2py` | bridge + often file I/O | `spm_smooth`, GLM, realign, segment, DARTEL | C · Tier 4 |

## 3. Methodology per class

- **Class A (nilearn, SimpleITK, ANTsPy)** — *kernel-vs-kernel, fair.* In-memory
  numpy round-trip, no I/O confound; a refs dep + a `numpy`-framework provider
  (`nilearn` / `sitk` / `ants`), identical in shape to the `sklearn` /
  `statsmodels` providers. The fp64 oracle and the gate are unchanged.
- **Class B (AFNI, FSL, FreeSurfer)** — *end-to-end-with-I/O-floor only.* No
  in-memory primitive API, so: measure the tool's wall time (read NIfTI →
  compute → write NIfTI), **separately** measure the NIfTI read+write floor at
  the same size, and report it labelled **"tool end-to-end (incl. mandatory
  I/O) vs nitrix in-memory op,"** with the I/O floor as context. Kernel-vs-
  kernel is impossible through the CLI — this is the honest framing, not a
  fudge. Requires **new infra** (see §4).
- **Class C (SPM)** — *language bridge.* `matlab.engine` (needs a MATLAB
  license) or `oct2py` + Octave (free, but SPM's Octave compatibility is
  partial). Bridge + MATLAB-JIT overhead on top of file I/O → lowest fidelity,
  highest friction.

## 4. Infrastructure implications

- **Class A** — *no new infra.* Add the lib to a refs dep group; register a
  `numpy`-framework provider (a distinct id for attribution, like `sklearn`).
  Lazy-import in the case if it is heavy (cf. `statsmodels`).
- **Class B** — *new "external-CLI provider" abstraction:* spawn a
  containerized binary on a temp NIfTI, parse its output, and a new
  **I/O-floor metric** so the compute estimate is `end-to-end − I/O-floor`. The
  binary suites cannot be vendored (license + multi-GB); they install via their
  official installers into a **container/pixi env** (DESIGN §7), a reproducible
  *host artifact* (like `tools/setup_refs_env.sh`), not committed. This is the
  larger lift and should start as a one-op feasibility spike (§5).
- **Class C** — a MATLAB/Octave bridge provider; heaviest, deferred.

## 5. Tiering & recommended sequence

1. **Tier 1 — nilearn + SimpleITK (now).** Highest value / lowest cost, both
   Class A, complementary coverage:
   - **nilearn** backfills refs we currently *lack or hand-roll*:
     `ConnectivityMeasure(kind='tangent')` for `tangent_project_spd` (we have
     **no** reference), `'partial correlation'` / `'precision'` for the
     precision family we just shipped, and `signal.clean` for
     `polynomial_detrend` / `residualise` / filters.
   - **SimpleITK** is the *industrial* floor for the morphology / smoothing /
     distance / resample / bias family (currently referenced only against
     `scipy.ndimage` + `cupyx`): a more credible bar than a scipy snippet, and
     the same ITK engine ANTs uses — closer to the raw primitive.
2. **Tier 2 — ANTsPy + dipy (the registration foils; shipped).** Both
   in-memory, pip-installable, Class A. ANTsPy (ITK engine): smoothing /
   N4-bias / transform / registration, the neuroimaging-level conveniences over
   SimpleITK. **dipy** (numpy / scipy / cython): the *second, independent*
   registration foil for the recipe family — `AffineRegistration` (rigid /
   affine, mutual information) and `SymmetricDiffeomorphicRegistration` (SyN on
   SSD, the log-Demons counterpart). Unlike ANTs' fixed internal schedule,
   dipy's pyramid is settable, so the recipe cases drive it with the *same*
   `(levels, iterations)` knob nitrix uses (the apples-to-apples per-config
   foil). Both run task-level (no shared oracle; recovery pinned in the tests).
3. **Tier 3 — external-CLI spike (FSL + AFNI).** Prove the end-to-end+I/O-floor
   methodology on *one* op each — **FSL `FLAME`** for `flame_two_level` and
   **AFNI `3dTproject`** for detrend — then generalize. This is where the
   **FLAME / 3dLME** references we could not obtain in-Python finally live.
4. **Tier 4 — FreeSurfer / Connectome Workbench / SPM (defer).** Revisit
   per-op-demand. Connectome Workbench is **boutique but the canonical bar** for
   CIFTI/grayordinate + surface ops (`-cifti-correlation` dense connectome,
   `-metric-smoothing`, `-surface-geodesic-distance`); it reuses the Class-B
   external-CLI infra from Tier 3, so it lands cheaply once that exists, when a
   nitrix surface / sphere / dense-connectome op needs the HCP-standard
   reference. (FS surface smoothing vs mesh-graph-conv; SPM GLM.)

## 6. What it backfills (the payoff)

Credibility — validating nitrix against the *actual tools* (not just numpy /
cupy) is the most persuasive comparison for a neuroimaging numerics library —
plus concrete reference gaps closed: **tangent** geometry (none today),
**partialcorr / precision** against the community-standard estimator,
**detrend / confound** against `signal.clean`, the **morphology / distance /
bias** family against ITK, and the **FLAME / 3dLME** LME references we flagged
as missing this sprint.

⚠️ **Nuance.** Where a Class-A tool merely wraps numpy / scipy / sklearn the way
we already do (e.g. parts of nilearn's connectome, SimpleITK filters that mirror
`scipy.ndimage`), the new reference is a *credibility label* more than a new
datapoint. The real added value is each tool's **distinct** implementation:
nilearn's tangent space + `clean` confound-regression + shrinkage estimators;
SimpleITK's ITK distance maps / bilateral / N4; the binaries' FLAME / 3dLME.

## 7. Verification discipline (match the right target)

Every tool needs the same **"match the right target"** check we applied to
statsmodels (flaky near boundaries) and Lomb-Scargle (stale normalisation): pin
the exact convention before trusting the reference. Known traps:

- **nilearn** `tangent` uses a specific reference-mean (geometric vs Euclidean)
  and a whitening convention; its connectome estimators default to **Ledoit-
  Wolf shrinkage**, not the raw inverse — a different definition from nitrix's
  `precision` (verify or pass `EmpiricalCovariance`).
- **SimpleITK** uses **(x, y, z)** axis order; `GetArrayFromImage` returns the
  reverse **(z, y, x)** — a silent transpose if unhandled. Filter conventions
  also differ (DiscreteGaussian takes *variance*, distance maps are
  signed / squared by flag).
- **dipy** registration is **task-level**, not op-vs-oracle: it converges to
  its *own* transform (a different optimum from nitrix / ANTs), so there is no
  shared oracle — recovery is pinned in the tests, not gated. Convention notes:
  its `AffineRegistration` metric is **mutual information** (so it differs from
  nitrix's SSD-driven GN/LM), `static`/`moving` map to **fixed**/**moving**
  (register moving→fixed, apply to moving), and `SymmetricDiffeomorphicRegistr-
  ation` takes its metric explicitly — we use **SSDMetric** as the log-Demons
  counterpart (dipy also offers CC/EM). Pyramid = `(level_iters, sigmas,
  factors)`, ordered **coarse→fine**, which we derive from the case's
  `(levels, iters)`.
- **AFNI `3dLME`** shells out to **R's `lme`/`nlme`** under the hood — its
  convergence and parameterisation are R's, not statsmodels'.
- **FSL FLAME** has its own MCMC / fixed-effects modes — match the mode
  (`flame1` vs `flame12`) to nitrix's estimator.
- **Connectome Workbench** operates on **grayordinates** (CIFTI structure =
  surface vertices + subcortical voxels); its surface smoothing is geodesic
  (along-mesh), not Euclidean, and `-cifti-correlation` has Fisher-z / covariance
  flags — match the grayordinate layout and the smoothing/normalisation
  convention before comparing to a nitrix surface/connectome op.

## 8. Cross-references

- [`COVERAGE_MANDATE.md`](COVERAGE_MANDATE.md) §2.2 (reference-strength
  taxonomy), §2.3 (CPU floor + strong on-target bar), §5 (separation of
  concerns).
- [`DESIGN.md`](DESIGN.md) §7 (isolated refs envs / pixi escape hatch).
- `nitrix/docs/feature-requests/perf-bench-feedback.md` (the consumer feedback
  channel; the Nyul-Udupa histogram-matching request that SimpleITK references).
