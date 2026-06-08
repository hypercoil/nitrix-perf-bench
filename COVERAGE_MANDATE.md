# nitrix-perf-bench — coverage audit & mandate

> **Status.** Decision-of-record for *what the suite measures* and *how
> coverage expands*, accepted 2026-05-28.  This is the coverage mandate;
> [`DESIGN.md`](DESIGN.md) is the architecture and
> [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md) is the row contract.
> It governs every case added from here, and the two ranked deficit lists it
> defines are the suite's primary feedback into `nitrix`.
>
> **Origin.** A coverage audit (findings in §1, measured from the committed
> store + `nitrix/docs/op_matrix.json`) found the architecture (P0–P3) built
> but coverage thin and skewed away from where engineering decisions are live.
> This document records the findings and the mandate they impose.

## 0. Why this document exists

The DESIGN delivered the *machinery* — rigorous async-correct measurement, the
fp64 fidelity-oracle ladder, multi-platform fan-out, the durable store,
regression gate, op-matrix feed, decision-input bundles, the HTML site (P0–P3).
What it did **not** yet deliver is *coverage*: the suite benchmarks a small,
skewed slice of `nitrix`'s surface. DESIGN §6's **P4 / nitrix BACKLOG B11** is
the migration that fills the gap one op at a time; this document is the
**organising principle** for that migration — so coverage grows toward the ops
and regimes where performance is actually a decision, not toward whatever is
easiest to port.

## 1. Audit findings (measured 2026-05-28)

> **These figures are a point-in-time snapshot, and the mandate exists to move
> them.** Treat the counts and ratios below as the *baseline this document was
> written to correct*, not as the current state — by construction they go stale
> (improve) as the migration proceeds. For the live picture, regenerate the
> **coverage-&-deficit report (§2.2)** against the current store + `op_matrix.
> json`; it is the source of truth, this section is the recorded starting line.
> The *shape* of the gap (§1.2) and the principles it motivates (§2) are the
> durable content; the exact numbers are evidence for them at one moment.

`nitrix` is a **GPU-first** substrate: Pallas-CUDA/TPU kernels with pure-JAX
fallbacks, an fp16/bf16/fp64/complex support matrix, and "measure on the
target" learned the hard way (the `ell_edge_aggregate` CPU↔GPU inversion —
DESIGN §6/P2). The audit measured the suite against that charter and found a
four-axis gap.

### 1.1 The four-axis coverage gap

| Axis | Measured state |
|---|---|
| **Breadth** | **11 of 59** catalogued ops have a perf-bench case (~19%). **26 ops have no perf characterisation anywhere** (capability-only). |
| **Platform depth** | Of the 11, **only 2** (`semiring_matmul`, `ell_edge_aggregate`) run on GPU. The **9 `PERF_AUDIT` ports are CPU-only**. Across all 59 ops the op-matrix carries exactly **one** real GPU ratio (`semiring_conv`, 1.78× vs cuDNN). |
| **Precision** | dtype is **never swept** — 124 stored rows, all f32 — though the schema has a `dtype` param field and the SPEC mandates a per-`(dtype, backend)` tolerance matrix. Reduced-precision perf *and* its fidelity tradeoff are wholly unmeasured. |
| **Reference quality** | The CPU references (numpy / scipy / sklearn, single-threaded) flatter nitrix 100–800×. They measure "XLA-compiled vs interpreted loop", not "is the kernel good". A *strong on-target* reference (cuDNN, torch-CUDA, cuSOLVER/CuPy, PyG-CUDA) exists for almost no op. |

### 1.2 Diagnosis — the suite is strongest where decisions are least live

The pattern across all four axes is the same: coverage is densest on **CPU
against weak references at f32**, the regime where nitrix wins trivially and no
kernel decision hangs on the number; it is sparse-to-absent on **GPU at scale,
reduced precision, and kernel-vs-kernel**, the regime where every live
question actually lives (B3 Pallas dispatch, B6 Gaussian kernel, B7 trilinear,
B10 LOG re-tune). A GPU library's GPU performance is nearly invisible to the
suite built to make it visible. **Correcting that inversion is the mandate.**

### 1.3 Structural note — the op-matrix perf columns are deprecated

B11's end-state makes `nitrix/docs/op_matrix.json` **capability-only** (jit /
grad / vmap / invariants); the perf columns are deleted and `nitrix/tools/
op_matrix.py::render_json` already emits none. So the durable feedback channel
**must live perf-bench-side**, not on `op_matrix_feed`. The op-matrix merge
remains a transitional, reviewed-copy convenience (DESIGN §4); the **coverage-&-
deficit report (§2.2)** is the channel that outlives the migration.

## 2. The mandate — six organising principles

### 2.1 Platform parity is the default, not a luxury
Every case is authored and run **multi-platform** (`--platforms
jax-cpu,jax-cuda12`, fanning to TPU / additional GPUs as hosts appear). A
CPU-only row is an *incomplete* case, not a finished one — the machinery for
this already exists (the store + `--render-from … --latest`); the gap is
execution discipline, made enforceable by the coverage gate (§4). GPU is the
deployment target; a number that does not exist on it does not inform a
decision.

### 2.2 The deficit is a first-class, reported artifact
A coverage-&-deficit renderer (L5) joins the op **catalogue**
(`op_matrix.json`, read purely as the list of ops) ⨝ the L4 store (by
`Case.op_qualname`) and emits, per op: **coverage status** (`unmeasured |
cpu_only | gpu_only | multiplatform`), **reference strength** (`floor_only |
strong_ref | internal_only`), **precision** (`f32_only | multi_dtype`), and the
**current ratio vs target** (§2.4). It then publishes **two ranked lists** — *(a)
unmeasured, by priority* and *(b) measured-but-lagging (over target), by
severity × consumer-traffic* — as markdown **and** machine-readable JSON for
the `nitrix` agent. This is the inverse of the op-matrix feed: the feed pushes
numbers we *have*; the deficit report surfaces the *gaps and
regressions-vs-target* — i.e. "the numerics most in need of improvement". It
obeys the renderer no-arithmetic rule (SCHEMA §G): selection + already-stored
ratios/targets only.

**Shipped (2026-05-29):** `report/coverage.py` + `tools/coverage_report.py`
(→ `reports/COVERAGE_DEFICIT.{md,json}`). "Lagging" is currently *slower than
the strong on-target ref* (per-op **targets**, §2.4, will refine the bar); a
`fast` run's ops are flagged **provisional**. First emission: 11/52 runtime ops
multiplatform, 9 with a strong GPU ref, **5 lagging on the L4** — ranked
`distance_transform` (~105×) ▸ `median_filter` (~5×) ▸ `spatial_transform`
(~1.8×) ▸ `erode`/`dilate` (~1.2–1.3×).

### 2.3 References must include a strong on-target bar
Keep the numpy / scipy / sklearn CPU references — they are the honest "what
you'd write without nitrix" **floor** that feeds the op-matrix — but every
decision-relevant op additionally carries a **strong reference on the
deployment platform**: cuDNN (conv), torch-CUDA (morphology / pooling /
interpolation), CuPy / cuSOLVER (cov / corr / residualise / eigh / symlog), PyG-
CUDA (edge-aggregate). These live in perf-bench's **isolated refs env** (DESIGN
§7; extend `tools/setup_refs_env.sh` for CuPy) so `nitrix` stays clean. Every
ratio is labelled with its reference strength, so a number that is green
against a weak floor is never mistaken for a win against the real bar.

**Floor-credibility upgrade (roadmap):** for a neuroimaging numerics library
the most decision-relevant *floor* is not a hand-rolled numpy snippet but *the
tool a practitioner actually runs*. [`DOMAIN_TOOL_BASELINES.md`](DOMAIN_TOOL_BASELINES.md)
plans adding nilearn / SimpleITK / ANTsPy (in-memory, kernel-vs-kernel) and the
binary suites FSL / AFNI / FreeSurfer / Connectome Workbench / SPM (end-to-end
with an I/O floor) as references — a credibility upgrade to this floor, *not* a
replacement for the strong on-target GPU bar above.

### 2.4 "Lagging" is defined by per-op targets, not eyeballed ratios
`ratio < 1 = faster` against a weak reference is almost always green and
therefore useless for triage. Each `Case` carries a typed, immutable
**target / budget** per platform (the decision bar: e.g. *within 2× of cuDNN*,
*beat naive-materialise*, *within 1.5× of PyG-CUDA*, *no worse than the JAX
fallback*). Targets are auto-seeded from first measurements then human-tuned.
The deficit report flags ops **over target**. This is the operational
definition of the suite's purpose — *identify functions lagging baselines and
targets.*

### 2.5 Precision is an axis where it pays
A declared **decision-tier dtype sub-sweep** (f16 / bf16 beside f32) for the
tensor-core-eligible and accumulation-sensitive ops (`semiring_matmul` REAL,
`semiring_conv`, the kernel / distance ops, covariance accumulation). The fp64
oracle already supplies truth and the gate already reads the fidelity
structure, so this exposes the perf↔fidelity tradeoff (e.g. the silent TF32
downgrade SCHEMA §C warns about) that f32-only coverage hides. Gated to ops
where reduced precision is realistic — not a blanket axis.

### 2.6 Scale is an axis: measure the curve, state the law *(provisional — evolving as the harder tier expands)*
A win at a small benched size says nothing at brain scale when the *asymptotics*
differ: a nitrix op can take a constant-factor GPU win on an algorithm with
worse FLOP or memory growth — more work, or more HBM, hidden at small `n` by GPU
parallelism — and lose, or **OOM**, before the size practitioners actually run.
This is **scale-gaming**: the size-axis analogue of the B18 dispatch-branch /
accuracy gaming, and the defence is symmetric.

- **Measure the curve, not a point.** A scalability-sensitive `Case` carries a
  brain-scale **size tier** (`large_param_points`: realistic single *and batched
  cohort* sizes), kept distinct from the small dev `param_points` so the
  `representative` (drift / dev anchor) stays fast. `--skip-large` drops the tier
  for dev cycles and stamps `coverage_mode=fast` (non-authoritative, exactly like
  `--skip-slow`); the authoritative sweep runs it. A single big "headline" size
  is **rejected** — it only relocates the gaming target; the curve + the
  crossover is the integrity surface.
- **State the cost law.** `Case.complexity` records the derived time + HBM
  asymptotic (nitrix vs the reference), so a crossover is *predictable from the
  algorithm*, not just observed at whichever sizes we happened to pick. (The EDT
  template's law also captures *why* the small-scale win exists — a low-depth,
  high-FLOP brute force beating a deeper low-FLOP scan while GPU wall-clock is
  depth-bound — so the crossover to flop/HBM-bound at scale is expected, not a
  surprise.)
- **Surface crossover, HBM growth, OOM-as-signal.** `tools/scaling_report.py`
  reads the store and reports, per op: the speed wins/losses across the sweep
  (the ratio is shape-dependent, *not* one crossover point — listed honestly,
  not forced into a false window), the HBM multiplier vs the baseline, a
  **projected-OOM headroom** from the per-element memory rate, and any point
  where nitrix OOMed while a baseline ran. An OOM / crossover before brain scale
  is a *reported outcome*, not a hidden row — the direct analogue of
  `ApproxBaseline` (accuracy-as-signal).

Brain-scale targets (per family, refined as the tier expands): volumetric 256³ +
a subject-cohort batch; surface / graph sparse `n`~40k–160k (fsaverage6/7), dense
`n`~400–1000 (parcellated); timeseries (10⁵ vertices × 10³ T).

*Status (2026-06-08): established on the `distance_transform` template (the size
tier, `complexity`, and `scaling_report.py`), then replicated to the morphology
family (2026-06-08), which **sharpened it**: an op can carry *both* a scaling
and a non-scaling path (the flat-box `reduce_window` scales — 256³ dilate wins
cupy 3.6× — while the explicit-SE im2col OOMs at 256³), so the tier must measure
*each dispatch path* and `scaling_report.py` keys rows by the SE (not just
shape) and projects OOM from the heaviest measured allocation. **This clause
stays living** — next is the eigensolver (sparse `n`~100k); expect the
brain-scale targets and headline metrics to keep sharpening, and amend here
rather than fork a parallel doctrine.*

## 3. Coverage taxonomy

- **Denominator.** ~7 catalogued entries are **host-side constructors**
  (`identity_grid`, `ell_from_dense`, `icosphere_*`, `mesh_k_ring_adjacency`) —
  NumPy construction, not runtime device ops. They are bucketed separately
  (an optional *construction-cost* track) so the runtime-perf denominator is
  **~52**, not 59. Holding a host-side constructor to the device-perf bar would
  be a category error.
- **Two tiers, one rigor floor** (reaffirming DESIGN §1). The **coverage tier**
  benchmarks every shipping runtime op at one declared representative point at
  the full rigor floor (fills the catalogue, drives list (a)). The **decision
  tier** layers parameter / dtype / reference sweeps only where a decision is
  live (drives list (b) and the decision-input bundles). Same measurement core;
  the difference is breadth, not rigor.

## 4. Prioritisation

Ranked by: on a real consumer's GPU hot path? · is perf *the* open engineering
question? · known/suspected deficit? · cheap mechanical port?

- **Tier 0 — platform-complete the 11 existing cases on GPU.** Pure
  orchestration; the single biggest jump in decision-relevant data. Stand up a
  **scheduled multi-platform store sweep** so the history / regression / trend
  machinery starts accumulating.
- **Tier 1 — kernel-decision ops** (perf *is* the question; decision-tier depth
  + strong ref + dtype): `semiring_conv` (cuDNN), `semiring_ell_matmul` (G0 /
  ELL), **`bilateral_gaussian`** (marquee §3.3, currently *zero* perf), the
  **pool / mesh family** (`mesh_pool_max` / `_unpool` / `_bary_upsample`,
  `max_pool_with_indices_nd` / `max_unpool_nd` — B2, Topofit / UNet GPU paths),
  and the standing watches **B10 LOG kernel** and **`distance_transform`** (the
  15–30× scipy gap, handled via the approximate / no-cross-impl oracle paths,
  SCHEMA §C).
- **Tier 2 — linalg / stats backbone** (high-traffic, strong refs exist):
  `symlog` / `symsqrt` / `sympower`, `linear_kernel` / `linear_distance` /
  `rbf_kernel`, `partialcov` / `precision`, `analytic_signal` / `hilbert` /
  `envelope`, `polynomial_detrend`, `tsconv`, `lme.reml_fit` /
  `flame_two_level`, interpolation / Lomb–Scargle.
- **Tier 3 — long tail, coverage-tier only**: `symmetric` / `sym2vec` /
  `vec2sym` / `toeplitz`, normalize / zscore / complex_decompose, geometry grid
  ops, `graph.laplacian` / `degree_vector` / `laplacian_eigenmap`, sphere ops.

The Tier-2/3 audit ports are stereotyped; a declarative `audit_case(...)`
builder (immutable `Case`, pure build fn, a `Protocol` ref adapter) keeps each
port to ~15 lines and consistent.

## 5. The feedback contract into nitrix

- **Deficit report** (markdown + JSON) committed to `reports/` — the primary
  channel; the `nitrix` agent reads the ranked lists.
- **HTML `/site`** dashboard — the human-facing overlay (perf section per op,
  capability matrix as overview).
- **op-matrix merge** — transitional, a reviewed *copy* only (the actual change
  is nitrix's own commit); retires when B11 strips the perf columns.
- **Separation of concerns (load-bearing).** The suite measures performance +
  the fidelity of its own comparisons only — correctness stays nitrix's test
  suite. `nitrix` never gains a perf-bench dependency. Capability stays
  nitrix's; only perf lives here.

## 6. Locked decisions (2026-05-28)

1. **Sequencing** — *platform-completion + the deficit report lead the first
   iteration* (run all 11 cases on GPU; ship the ranked gap report) over
   decision-depth-first or mechanical-breadth-first.
2. **GPU references** — *build the strong on-target GPU refs now* (CuPy /
   cuSOLVER + torch-CUDA in the isolated refs env), rather than relying on
   internal-only or the flattering CPU floor.
3. **"Lagging" definition** — *per-op typed targets on the `Case`* (auto-seeded,
   human-tuned), rather than auto-derived rules only or ratios without targets.

## 7. Sequencing

- **A — orchestration.** GPU-complete the 11 cases; scheduled store sweep; first
  coverage-&-deficit report. *No new cases.*
- **B — breadth.** `audit_case` builder; Tier-2 backbone with CPU floor + GPU
  strong refs; CuPy added to the refs env.
- **C — depth.** Tier-1 kernel ops with on-target refs, dtype sub-sweeps, per-op
  targets; promote the B10 / distance / edge-aggregate watches.
- **D — process.** Coverage CI gate (a migrated op must have a case; a case must
  not silently lose a platform); regression cadence; a *ships-with-a-case* SLA
  so coverage tracks the catalogue as `nitrix` grows (SPEC v0.3 §12) — and, for
  a **scalability-sensitive op**, *ships-with-a-scalability-case*: a perf win on
  such an op lands with its size tier (`large_param_points`) + cost law
  (`Case.complexity`), so the win is certified at brain scale, not just at a
  small benched size (§2.6).

**Shipped (2026-05-29) — the slow-baseline dev-cycle guard.** Sprints run a
fast inner loop / comprehensive outer loop: skip known-slow benchmarks during
iteration, run the full matrix at sprint end. `Case.slow_baselines` declares the
known-slow baselines (each with an evidence-based, *device-stamped* reason —
e.g. `semiring_matmul`'s `naive-dense`, whose ~432 s (L4) / ~580 s (A10G) cold
compile at 512³ log is GPU-specific, ~0.3 s on CPU). `--skip-slow` drops them,
recording each as a `slow_skipped` row and stamping the run `coverage_mode=fast`.
The load-bearing guard against "fast by default → slow ops never measured": the
**op_matrix feed ignores `fast` rows** (only a full run blesses the matrix) and
the **regression gate treats a deliberately-skipped current row as an omission,
not a regression** (so a fast run can detect regressions on its subset without
false-failing the skipped slow baselines). Authoritative coverage = a full
sweep (omit `--skip-slow`).

**Shipped (2026-05-29) — Phase B: the GPU bar for the audit ops.** All 9 audit
cases now carry a **CuPy on-target GPU reference** (`cupy.corrcoef`,
`cupy.linalg.lstsq`/cuSOLVER, `cupyx.scipy.ndimage.*`) in an isolated
`refs-cupy` env, gated GPU-only via `Provider.requires='gpu'` + a runner
platform-applicability skip (`platform_not_applicable` off-GPU — critical
because CuPy ignores `JAX_PLATFORMS`). The apples-to-apples `jax-cuda12` ratio
replaces the misleading nitrix-GPU-vs-CPU-floor view and immediately surfaced
**on-target deficits the floor hid**: nitrix wins on cov/corr/residualise
(2–29×) and gaussian (~2×) but **lags CuPy** on erode/dilate (~1.2×),
spatial_transform (~1.8×), median_filter (~2–5×), and distance_transform
(~9–100×, the iterative-tropical vs exact-EDT mismatch) — plus a memory signal
(nitrix morphology ~34–72 MB HBM vs CuPy <0.3 MB). These are exactly the
"performance lagging on the deployment target" ops the mandate exists to find;
they feed the nitrix agent as Pallas-kernel / algorithm candidates. *Still open
in Phase B:* the `audit_case` builder and the Tier-2 backbone (new ops);
torch-CUDA refs for conv/pooling are Phase C.

**Shipped (2026-05-29) — Tier-2 breadth begins: the linalg eigh-family.**
`symlog` / `symsqrt` / `sympower` ported (nitrix-jax + `scipy.linalg` CPU floor
+ a CuPy GPU ref + an fp64 eigh oracle). A measure-on-the-target finding (and a
caution-driven deep-dive): **dense cuSOLVER `eigh` is broken at d≥256 on this
L4 / driver-580** — *both* `cupy.linalg.eigh` and a *bare* `jnp.linalg.eigh`
(eager and jitted) fail; d=64 works (the documented jaxlib/cuSOLVER class of
issue). **But nitrix's matrix-function ops run on the GPU honestly** because
they *consume* the decomposition into `f(A)=V·diag(f(λ))·Vᵀ`, which XLA lowers
off the broken cuSOLVER path (verified correct on a provably cuda-only process;
tested to be the *consumed-eigh* lowering, **not** `safe_eigh`). So nitrix does
GPU eigh at sizes where CuPy/bare-eigh can't — a real, scoped win. **Caveat:**
it does **not** extend to ops that *return* eigenpairs (e.g.
`graph.laplacian_eigenmap`), which would still hit the cuSOLVER failure on GPU.
Infra: runner classification `gpu_solver_unavailable` (genuine cuSOLVER) /
`backend_unavailable`; a reason-robust `gpu_blocked` report flag (dormant for
this family, since nitrix runs). The CuPy ref's d≥256 failures are recorded
(`gpu_solver_unavailable`), so the apples-to-apples GPU bar holds at d=64.

**Shipped (2026-05-29) — Tier-2 stats/signal: the Hilbert family.**
`analytic_signal` / `hilbert_transform` / `envelope` ported (nitrix-jax +
`scipy.signal.hilbert` CPU floor + `cupyx.scipy.signal.hilbert` GPU ref + fp64
scipy oracle). FFT-based, so — unlike the eigh family — **GPU-pure** (no
cuSOLVER): a clean apples-to-apples GPU bar at every size. Finding: nitrix is
**≈ CuPy on the L4** (par to ~13% faster — both FFT-bound) and ~1.6–2.3× faster
than scipy on CPU; nitrix's `analytic_signal` carries ~7× the HBM of CuPy's (a
materialisation note). Added **complex-aware fidelity** (compare via
`|out − ref|`) for `analytic_signal`'s complex output — also unblocks
`complex_decompose`. Coverage: **17/52 runtime ops multiplatform, 15 with a
strong GPU ref**.

**Shipped (2026-06-08) — the scale tier (the scale-gaming guard, §2.6).** After
B18's four case-hardening wins closed the dispatch-branch + accuracy axes, the
**size axis** opened the same hole: several wins (EDT, morphology disk/ball,
the dense eigensolver backward) are constant-factor GPU wins on a worse
asymptotic / memory growth that loses — or OOMs — before brain scale. The
substrate: `Case.large_param_points` (a brain-scale + *batched cohort* size
tier, run by default; `--skip-large` drops it + stamps `coverage_mode=fast`,
verified), `Case.complexity` (the derived cost law), and
`tools/scaling_report.py` → `reports/SCALING.md` (per-op curve: speed
wins/losses, HBM multiplier, projected-OOM headroom, OOM-as-signal).
**Template = `distance_transform`** (L4): speed is *competitive* (nitrix wins
5/10 sizes; CuPy 1.1–1.7× ahead at several, nitrix ahead at the largest batched)
but the real scale risk is **HBM — nitrix is a 5–2051× memory multiplier**
(33.6 MB @64² vs CuPy ~0; 671 MB @16×128³ vs 134 MB) → **~5× less OOM headroom**
(~1200 vs ~6000 Melem projected), and the batched-cohort regime is where it
bites. The min-plus matmul does *more FLOPs but in one shallow pass* vs F-H's
deeper sequential scan — the working hypothesis for why it wins while GPU
wall-clock is depth-bound at small scale and loses once flop/HBM-bound; the
semiring substrate is differentiable, but that is a *bonus*, not why it was
chosen. *Next:* replicate the tier to morphology (the OOM exemplar) and the
eigensolver (sparse `n`~100k), refining §2.6 as the learnings land.

**Shipped (2026-06-08) — scale tier replicated to morphology (the OOM
exemplar).** `erode` / `dilate` / `open` / `close` gain the brain-scale tier
(256³ box / ball + a 4×128³ cohort) via the shared `build_morph_large` (nitrix +
the cupy GPU ref only, no fp64 oracle — correctness is pinned at the dev tier;
this tier measures *scale*). The headline (L4): the **flat box scales** — 256³
`dilate` wins cupy **3.6×** at 3× HBM — while the **explicit-SE (disk/ball)
im2col does not**: 256³ ball r2 is **~300× slower + ~17 GB HBM**, and ball r4
256³ **OOMs (~49 GB)** while cupy runs (~12 ms) — the genuine OOM EDT lacked.
~253× less projected OOM headroom; OOM-as-signal fires for all four ops. So an
op can carry *both* a scaling and a non-scaling dispatch path (the §2.6
refinement). This expansion also **sharpened `scaling_report.py`**: rows are
keyed by the SE (not just shape — several SEs share a grid size, and collapsing
them hid the fast box behind the slow ball), and the projected OOM is taken from
the *heaviest measured allocation*'s per-element rate (a small point's rate is
inflated by fixed allocator overhead). *Next:* the eigensolver (sparse
`n`~100k, the O(n²) dense backward).

## 8. Cross-references

- [`DOMAIN_TOOL_BASELINES.md`](DOMAIN_TOOL_BASELINES.md) — roadmap for the
  canonical-domain-tool references that upgrade the §2.3 CPU floor (nilearn /
  SimpleITK / ANTsPy / FSL / AFNI / FreeSurfer / Connectome Workbench / SPM).
- [`DESIGN.md`](DESIGN.md) §1 (two-tier coverage), §4 (feedback loop), §6/P4
  (the migration this mandate steers).
- [`SCHEMA_AND_LIFECYCLE.md`](SCHEMA_AND_LIFECYCLE.md) §A (row + `dtype` field),
  §C (oracle ladder + no-cross-impl / approximate oracle), §F (gate), §G
  (renderer no-arithmetic).
- `nitrix/BACKLOG.md` B11 (the migration tracker), B2 / B3 / B6 / B7 / B10 (the
  live kernel decisions this coverage must inform).
- `nitrix/docs/op_matrix.json` — the op catalogue the deficit report joins
  against.
