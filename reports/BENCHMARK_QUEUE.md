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
> priority). Also fixed `scaling_report` to size `d`/`b` params. **Remaining:**
> `max_pool_with_indices_nd` / `max_unpool_nd` (pooling).

| op | ref strategy | discipline notes |
|---|---|---|
| `morphology.distance_transform_edt` | `scipy.ndimage.distance_transform_edt` (oracle) + cupyx | sibling of the benched `distance_transform` EDT exemplar; **scale tier already templated** (depth-vs-FLOPs crossover, HBM hog) |
| `morphology.connected_components` | `scipy.ndimage.label` (oracle) + cupyx label | **iterative label-propagation** — interesting GPU scaling; iterative-op caveat; pin label-permutation invariance |
| `morphology.largest_connected_component` | scipy.ndimage + argmax (oracle) | composes the above |
| `morphology.max_pool_with_indices_nd` / `max_unpool_nd` | torch maxpool / scipy (oracle) + cupy | pooling + the index round-trip; warranted: index/argmax-tie convention |

## 3. Stats

| op | ref strategy | discipline notes |
|---|---|---|
| `stats.pca_transform` / `pca_fit` / `pca_inverse_transform` | `sklearn.decomposition.PCA` (floor) + numpy SVD (oracle) + cupy | **sign/component-order ambiguity** (eigenvector sign — cf. the ARPACK flake we just fixed): use a sign-robust comparison; pin the centring/whitening convention |
| `stats.conditionalcorr` / `conditionalcov` | numpy exact (oracle) + nilearn | siblings of the benched precision/partialcorr family; warranted: the exact conditioning estimator |
| `stats.pairedcorr` / `pairedcov` | numpy exact (oracle) | warranted: paired vs pooled definition |

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
