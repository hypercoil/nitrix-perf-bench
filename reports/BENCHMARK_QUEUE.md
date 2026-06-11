# Benchmark queue — next batches (registration · morphology · stats · metrics)

Hand-authored plan (not the generated `COVERAGE_DEFICIT.md`). The op-matrix
re-inventory (2026-06-10) found **147 of 225 ops unbenched**; this queues the
high-value compute ops in the four families requested, each to ship **with the
full discipline** (checklist below). Triage tiers are in the session notes; this
is the Tier-A worklist.

## The discipline every queued case must meet

1. **B18 anti-gaming** — measure the *dispatch branch users actually hit*, at
   *pinned accuracy*, *called the way users call it* (not a fast degenerate
   path). Pin the accuracy contract in a test.
2. **Warranted comparison** (DOMAIN_TOOL_BASELINES §7) — verify the reference
   computes nitrix's *exact* quantity before trusting it; record the convention
   (and any deliberate divergence as a labelled `ApproxBaseline`, reported not
   gated).
3. **Oracle ladder** — fp64 reference where one exists (every baseline scored
   against it); **task-level** (`fp64_reference=None`, recovery/accuracy pinned
   in tests) for iterative drivers with no shared optimum.
4. **References** — numpy/scipy/sklearn **CPU floor** + a **strong on-target GPU
   bar** (cupy / torch-CUDA) + a **domain tool** where one exists (ITK / ANTs /
   dipy / nilearn).
5. **Scale tier** (COVERAGE_MANDATE §2.6) — `large_param_points` (brain-scale /
   batched) + a stated `complexity` cost law; certify the win holds at scale via
   `scaling_report`. Watch the **peak_hbm cold-measurement caveat**
   (`reports/REGISTRATION_SCALING.md` §4) for compile-heavy ops.
6. **Drift** — seed the `op_drift_manifest` (signature + representative
   behaviour); re-bless only intentional changes.
7. **Iterative-op caveat** (new, `reports/REGISTRATION_SCALING.md` §early-stop) —
   for any *iterative* op, note whether nitrix runs a **fixed** step count
   (`lax.scan`) vs a reference that **early-exits** on convergence; prefer an
   **iso-accuracy** (time-to-target) read so wall-clock isn't conflated with
   iteration count.

---

## 1. Registration (continue the current thrust)

> **§1 COMPLETE (2026-06-10).** ✅ penalties (`gradient_smoothness`,
> `bending_energy`, `jacobian_folding_penalty`) — `63b145a`; ✅ transform-exps
> (`rigid_exp`, `affine_exp`, `rigid_log`) — `0c5dae1`; ✅ field algebra
> (`spatial_gradient`; `invert_displacement` — the iterative + IFT-diff
> unique-win, with `lax.while_loop` early-exit done right; `compose_velocity`
> BCH order-2). All with warranted oracles + scale/batch tiers + drift seeds.
> **Next: §2 Morphology.**
>
> **§1 v2 — registration-suite-v2 recipes (2026-06-11).** ✅ `greedy_syn_register`
> (SyN vs ANTs `SyNOnly` + dipy), `volreg` (motion realignment vs ANTs
> `motion_correction`), `bbr_register` (nitrix-only — no ITK/ANTs BBR), plus
> **cross-grid** rigid/affine (`WorldSpace`, different shape + anisotropic
> spacing) and **anisotropic** demons/SyN (`spacing=[1,1,3]`), added alongside
> the shared-grid points. Drove the **economic verdict** tooling
> (`tools/economic_report.py` → `ECONOMIC.md`): the nitrix-GPU win counts only
> if **multiplicative** over the ~4× GPU:CPU hardware premium. Findings
> (`REGISTRATION_SCALING.md` §8): the genuine multiplicative wins are
> **volreg** (the batching story, 54→94× amortized, grows with T) and **demons**
> (29× vs ITK); `syn`/`rigid`/`affine` frequently **fail** the 4× bar at brain
> scale because ANTs is fast C, and single-run is compile-dominated everywhere
> but volreg T=500. **Caveats:** ANTs is *not* the community moco standard
> (AFNI `3dvolreg` / FSL `mcflirt`) nor a BBR tool (FSL/FreeSurfer) — a planned
> `/scratch` install will re-bench against the real community tools.
>
> **§1 v3 — registration-suite-v3 re-bench (2026-06-11).** ✅ Re-measured the whole
> registration surface against nitrix v3 (`356c768`: Force-protocol SyN + perf
> levers), the **installed community tools** (AFNI `3dvolreg` / FSL `mcflirt` on
> `/scratch`, see README; `setup_neuro_refs.sh` is the reproducible recipe), and
> **real anatomy** (MNI152 T1 planted-warp points in rigid/affine/demons/syn).
> Full authoritative matrix (`coverage_mode=full`, both platforms). The numbers
> **superseding the v2 note above** (`ECONOMIC.md` / `REGISTRATION_SCALING.md` §8):
> **volreg** now bars against FSL `mcflirt` **I/O-floor-subtracted** (`3dcalc` /
> `fslmaths` no-ops measure the NIfTI round-trip) → an **honest 7.8–12.5×
> amortized**, *not* the 54→94× the slow-ANTs bar inflated; ANTs
> `motion_correction` **timed out at T=500** (confirming it is not the moco bar).
> The v3 perf levers lift **rigid 6/7** and **affine 5/7** favorable (was 1/6, 2/6);
> **demons 6/6** (vs SimpleITK), **bbr** 29.7× @N=80 000. The cross-grid
> (`WorldSpace`) points and **SyN 128³** land *not multiplicative enough* (the
> honest scale finding); **single-run is favorable nowhere** (compile-bound). Three
> honest CPU timeout rows (dipy demons, ANTs moco T=500, nitrix-CPU SyN 128³) —
> none break a verdict. **Still open:** FreeSurfer `bbregister` for a BBR domain bar.

| op | ref strategy | discipline notes |
|---|---|---|
| `register.bending_energy` | numpy exact reimpl (oracle) + cupy | stencil over a displacement field; **scale tier** (bandwidth, brain-scale); differentiable (training penalty) |
| `register.gradient_smoothness` | numpy exact + cupy | as above |
| `register.jacobian_folding_penalty` | numpy exact (Jacobian-det of φ) + cupy | warranted: pin the folding/Jacobian convention |
| `geometry.affine_exp` / `rigid_exp` | `scipy.linalg.expm` (oracle) + cupy | siblings of the benched `matrix_exp`; **batch tier** (many transforms); per-call overhead story |
| `geometry.rigid_log` | `scipy.linalg.logm` (oracle) | the inverse map; convention (branch) trap |
| `geometry.spatial_gradient` | numpy/scipy gradient (oracle) + cupy | the Demons building block; bandwidth, scale tier |
| `geometry.invert_displacement` | numpy fixed-point (oracle) | **iterative + IFT-differentiable** — a likely *unique-win* (cf. the eigensolver: the adjoint is the win); apply the iterative-op caveat |
| `geometry.compose_velocity` (BCH) | numpy BCH reimpl (oracle) + cupy | elementwise/stencil; order-`k` BCH branch |

## 2. Morphology

> **Progress (2026-06-10):** ✅ distance/connectivity (`distance_transform_edt`,
> `connected_components`, `largest_connected_component`). **Finding (filed on
> nitrix `main`, `ce04d7c`):** the connectivity ops scale *poorly* vs cupyx
> `label` (~2×→18× behind by 160³ — a kernel/algorithm scale risk); the EDT is
> the semiring euclidean alias (wins small / loses large — the known
> depth-vs-FLOP trade-off; a scale-aware semiring↔F-H dispatch was filed lower-
> priority). Also fixed `scaling_report` to size `d`/`b` params.
> ✅ pooling (`max_pool_with_indices_nd`, `max_unpool_nd`) — a clean nitrix-GPU
> win (1.6–1.9× over cupy; the with-indices argmax is ~2.6× a max-only pool,
> measured with a DCE guard). **§2 COMPLETE. §3 Stats: PCA family landed.**

| op | ref strategy | discipline notes |
|---|---|---|
| `morphology.distance_transform_edt` | `scipy.ndimage.distance_transform_edt` (oracle) + cupyx | sibling of the benched `distance_transform` EDT exemplar; **scale tier already templated** (depth-vs-FLOPs crossover, HBM hog) |
| `morphology.connected_components` | `scipy.ndimage.label` (oracle) + cupyx label | **iterative label-propagation** — interesting GPU scaling; iterative-op caveat; pin label-permutation invariance |
| `morphology.largest_connected_component` | scipy.ndimage + argmax (oracle) | composes the above |
| `morphology.max_pool_with_indices_nd` / `max_unpool_nd` | torch maxpool / scipy (oracle) + cupy | pooling + the index round-trip; warranted: index/argmax-tie convention |

## 3. Stats

> **Progress (2026-06-10):** ✅ PCA family (`pca_fit`, `pca_transform`,
> `pca_inverse_transform`). Sign/rotation ambiguity dodged by scoring the
> invariant quantity: `pca_fit` gates on `explained_variance` (the unique top-k
> covariance eigenvalues, not the ±/rotation-ambiguous components);
> transform/inverse share one fixed pre-fitted basis across all frameworks, so
> their matmul output is unambiguous. **Finding (re-measured, corrects
> [[perfbench-gpu-eigh-blocker]]):** the older "`safe_eigh` routes to CPU at
> d≥256" assumption did **not** reproduce — in a fresh worker the cuSOLVER eigh
> stays GPU-native through d=2048 (nitrix 44.9 ms vs cupy device-eigh 41.6 ms:
> parity), the CPU fallback a latent net that only fires on handle-creation
> failure in long-lived/pressured contexts. So `pca_fit` is GPU-parity with
> cupy + a 6–12× CPU win over sklearn's full-SVD (`sklearn.PCA` made a
> `slow_baseline` — times out at d=2048). The matmul twins: `pca_transform`
> parity with cupy; `pca_inverse_transform` beats cupy 1.8–2.7× at scale (it
> consumes the small `Z (n,k)`, not the full `X (n,d)`).
>
> **Progress (2026-06-10, cont.):** ✅ paired / conditional family
> (`pairedcov`, `pairedcorr`, `conditionalcov`, `conditionalcorr`). Conventions
> matched to ~1e-16 vs the jitted op (`ddof=1`, `rowvar=True`, no-intercept
> residualise, the `+eps`-outside-sqrt corr norms); numpy fp64 oracle + cupy
> GPU twin (no nilearn — it has no cross-/conditional-cov kind, and
> `signal.clean` adds an intercept + detrend → a different estimator). Paired
> are pure BLAS; conditional residualise on a **tiny `(d,d)` confound Gram**
> (Cholesky), so matmul-bound and GPU-robust — the `(d,d)` solver ran
> GPU-native (contrast `pca_fit`'s `(d,d)` eigh at parcel `d`). `conditionalcov`
> wins 5/5 sizes vs cupy at scale. **Finding (filed on nitrix `main`,
> `c865f67`):** `pairedcorr` forms the **full** `cov(X)`/`cov(Y)` just to read
> their diagonals — ~3× matmul; the direct-variance ref is ~2× faster from
> c≳512 on CPU+GPU (the `pairedcov` control is at parity, isolating the cost
> to the redundant covs; flagged a scale risk in `SCALING.md`). **§3 Stats
> COMPLETE.**

| op | ref strategy | discipline notes |
|---|---|---|
| `stats.pca_transform` / `pca_fit` / `pca_inverse_transform` | `sklearn.decomposition.PCA` (floor) + numpy SVD (oracle) + cupy | ✅ **DONE.** sign/rotation ambiguity → score `explained_variance` (fit) + a fixed shared basis (transform/inverse); eigh measured GPU-native to d=2048 |
| `stats.conditionalcorr` / `conditionalcov` | numpy exact (oracle) + cupy | ✅ **DONE.** residualise (tiny `(d,d)` Gram, GPU-robust) + cov; numpy oracle (no nilearn — different estimator); `conditionalcov` wins 5/5 vs cupy at scale |
| `stats.pairedcorr` / `pairedcov` | numpy exact (oracle) + cupy | ✅ **DONE.** pure-BLAS cross-blocks; `pairedcov` at parity, `pairedcorr` ~2× behind from redundant full covs (nitrix FR `c865f67`) |

## 4. Metrics

| op | ref strategy | discipline notes |
|---|---|---|
| `metrics.joint_histogram` | `numpy.histogram2d` (hard-bin oracle) + cupy | **soft vs hard binning** — nitrix's is the differentiable soft-Parzen hist (the MI-case divergence, already documented): hard-bin numpy is a labelled `ApproxBaseline`, soft-bin reimpl is the oracle |
| `metrics.dice` / `jaccard` | numpy hard overlap (oracle) + monai/torch | **soft (differentiable) vs hard** overlap convention; pin it |
| `metrics.info_nce` / `koleo` / `dino_cross_entropy` / `ibot_cross_entropy` | torch reference impls (oracle) | SSL/contrastive losses — **training-relevant (GPU)**; warranted: temperature / normalisation / centring conventions are subtle, verify each against the source formula |

---

## Sequencing

1. **Registration regularisers + transform-exps** (§1) — closes the registration
   surface we are already deep in; several have the bandwidth/scale and
   unique-win (`invert_displacement`) angles.
2. **Morphology** (§2) — `distance_transform_edt` rides the existing EDT scale
   template; `connected_components` adds the iterative-label GPU story.
3. **Stats PCA + connectivity** (§3) — reuses the precision/partialcorr
   machinery; the PCA sign-robustness is the one new wrinkle.
4. **Metrics** (§4) — `joint_histogram` first (the MI block, convention already
   mapped), then the overlap + SSL losses.

Each batch ships its cases + tests + drift seed + scale tier, then a
`scaling_report` pass, before the next batch — the established sprint cadence
(skip known-slow during the sprint, full matrix at sprint end).
