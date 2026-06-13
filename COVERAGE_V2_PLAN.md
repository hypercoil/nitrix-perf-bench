# COVERAGE v2 — multi-axis coverage plan

Status: **Phases 1–3 landed.** Phase 1 = scale / economic / real-data axes +
the community-gold ref-class fix (pure store inference). Phase 2 = `Case.tier`,
the tier-gated **marquee coverage matrix** + completeness score + the
*marquee-unmet* deficit, plus **orphan-case surfacing** (benchmarked cases
absent from a stale `op_matrix.json`). Phase 3 = the **full coverage matrix**
(every op with a case, worst-vs-tier first) + the COVERAGE_MANDATE §2.2 update;
and a score refinement — the **economic verdict is an indicator/result, a
matrix column + its own deficit, but NOT in the completeness score** (a
not-multiplicative op is a finding, not a coverage gap). Phase 4 (optional
coverage gate) remains.

Extends the coverage-&-deficit report
(`report/coverage.py`, `tools/coverage_report.py`; COVERAGE_MANDATE §2.2) from a
3-axis record to a **tier-gated coverage matrix**, surfacing — as first-class
information — realistically-large cases, the economic indicator (GPU as a
multiple of CPU), and (for marquee ops) tests on real brain data against real
community baselines.

**Finding (Phase 2): the catalogue is stale.** Three marquee registration ops —
`volreg`, `bbr_register`, `greedy_syn_register` — are benchmarked but absent
from nitrix's `op_matrix.json`, so the join cannot see them. Surfaced as
*orphan cases*; the fix is to regenerate the catalogue in nitrix.

## Design principles
1. **intent × evidence** — each axis pairs a *declaration* on the `Case` with
   *evidence* in the store. A gap is intent without evidence.
2. **tier-gated applicability** — a `tier` per op sets which axes are *required*;
   the score is `satisfied / applicable`, not `/ N`. (A utility op isn't dinged
   for lacking real-brain data; a marquee one is.)
3. **factor, don't duplicate** — the economic verdict and the scale/OOM + size
   logic already live in the `tools/*_report.py` scripts; lift their cores into
   `report/` modules that both the tools and `coverage.py` import. No metric
   arithmetic is added (every figure read from L1 rows; SCHEMA §G).

## Locked decisions (2026-06-12)
1. **`tier` = 2 levels: `standard | marquee`.** The tier's only job is to make
   real(istic) tests required (marquee) or not (standard).
2. **`tier` lives on the `Case`** (benchmarking *policy*); `op_matrix.json` stays
   capability-only (nitrix/perf-bench separation).
3. **Input realism = a 3-rung ladder: `synthetic | real_planted | real_full`.**
   `real_planted` = real DATA with a synthetic/known ground truth (a planted warp
   on a real image — recoverable truth; the registration MNI152 harness);
   `real_full` = full realism, the ACTUAL problem (real data, no planted truth).
   Explicit "hard synthetic" flagging is still **deferred** (subjective / ill-posed
   across cases). The rung is inferred from a point's `data`/`regime` tag — no
   per-case "hardness" declaration this round.
4. **Economic verdict point = the largest real/large measured point** where
   available, else the **representative** point **with a `not-authoritative`
   marker**. The full size-tier economic detail stays in `ECONOMIC.md`.

## The axes (target: 7)

| axis | declared by | evidenced by | states |
|---|---|---|---|
| platform *(exists)* | — | ok nitrix rows/platform | cpu / gpu / multi / unmeasured |
| gpu-ref *(exists)* | — | strong on-target ref (cupy/torch) ratio | none / internal / floor / strong |
| precision *(exists)* | — | dtypes measured | unmeasured / f32_only / multi_dtype |
| **scale** | `large_param_points` | ok nitrix row at the largest large-point | no_tier / declared / scaled / **scale_capped** |
| **economic** | — (applies iff nitrix on GPU) | shared economic join vs cost-multiple | favorable / amortized_only / not_multiplicative / n/a / unmeasured (+`authoritative` flag) |
| **input realism** | point `data`/`regime` tag | max realism over ok nitrix rows | synthetic / real_planted / real_full |
| **domain ref** | — | best community-gold baseline (ants/fsl/…) + ran-on-real | none / present (+`on_real`) |
| **tier** *(Phase 2)* | `Case.tier` | — | standard / marquee — gates which axes are *required* |

## Data-model changes
- **`Case`** (`cases/_base.py`): add `tier: str = 'standard'` (Phase 2). No
  realism field needed — realism is inferred from point tags (decision 3).
- **param-point realism convention**: a point is **real** iff it carries a
  real-dataset marker — `'data'` present (e.g. `{'data':'mni152'}`, already used
  by the registration cases) or `'regime':'real'`. Else **synthetic**.
- **`OpCoverage`** (`report/coverage.py`): add `scale_status`, `largest_ok_size`,
  `scale_cap_reason`, `economic_verdict`, `economic_amortized`,
  `economic_authoritative`, `input_realism`, `domain_ref`, `domain_ref_on_real`,
  `tier`, `coverage_score: (satisfied, applicable)`.

## Shared modules (the refactor)
- **`report/economic.py`** (new): `COST_MULTIPLE`, `verdict(amortized, single, bar)`,
  `analyse(case, rows, bar) -> [per-point]` — lifted from `tools/economic_report.py`
  (`_GPU_CPU_COST_MULTIPLE`/`_verdict`/`_analyse`). The tool becomes a thin renderer.
- **`report/sizing.py`** (new): `size_elems(param)`, `label(param)` — lifted from
  `tools/scaling_report.py` (`_size_elems`/`_label`). The tool imports them.
- **`report/coverage.py`** imports both; `tools/{economic,scaling}_report.py`
  re-export for back-compat. Behavior of ECONOMIC.md / SCALING.md must be
  byte-identical after the refactor (verified by regenerating + diffing).

## Ref-class fix (the latent bug)
`_ref_class` only counts cupy/torch as `strong`, so every community gold standard
(ANTs/FSL/AFNI/FreeSurfer/dipy/SimpleITK/statsmodels) is currently classified
`internal`/`floor` and is invisible to the "strong ref" count. Add a **`domain`**
class keyed on the **baseline-name namespace** (robust — the CLI tools are all
`framework='numpy'` but named `ns.tool`), excluding `*.iofloor`:
`_DOMAIN_NS = {ants, fsl, afni, freesurfer, dipy, simpleitk, statsmodels}`.

## Classifiers (added to `build_coverage`)
Widen `_op_to_case()` to map `op -> Case` (it already iterates `CASES`). Per op:
- `_scale_status(case, rows)` — via `sizing.size_elems` over `large_param_points`:
  nitrix ok at largest ⇒ `scaled`; oom/timeout at a large point w/ a smaller ok ⇒
  `scale_capped` (+reason); none measured ⇒ `declared`; no tier ⇒ `no_tier`.
- `_economic(case, rows)` — `economic.analyse(...)`; pick the verdict point per
  decision 4 (largest real/large else representative+`authoritative=False`).
- `_input_realism(rows)` — `real` if any ok nitrix row is on a real point, else
  `synthetic`.
- `_domain_ref(rows)` — best `domain`-class ok ref + whether it ran on a real point.
- `coverage_score` (Phase 2) — satisfied / applicable among the tier's required axes.

## Rendering
1. **`## Coverage matrix`** — one row/op: `op | platform | scale | economic |
   input | gpu-ref | domain-ref | score`, glyphs `✓/✗/⚠/n-a`, sorted worst-vs-tier
   first.
2. **Per-axis deficit sections** (ranked): *Scale-fragile* (`scale_capped`),
   *No economic win* (`not_multiplicative`), *Synthetic-only* (marquee w/o real),
   *Marquee unmet* (Phase 2). Existing "lagging"/"under-covered"/"GPU-blocked"
   sections stay.
3. **JSON**: new fields + `coverage_score` + a `by_axis` deficit index; bump a
   `schema` tag.

## Phases
- **Phase 1 — pure store inference (no `Case` annotation needed):** the `domain`
  ref-class fix, the `report/economic.py` + `report/sizing.py` refactors, and the
  **scale**, **economic**, **input-realism**, **domain-ref** axes + their deficit
  sections. Regenerate; reclassifies this session's work (flame → scale_capped +
  economic n/a + domain_ref fsl.flameo; registration → domain_ref ants/fsl + real).
- **Phase 2 — declarations:** add `Case.tier`; tier-gating + `coverage_score` +
  the *Marquee unmet* deficit + the matrix sort; annotate the marquee set.
- **Phase 3 — matrix polish + COVERAGE_MANDATE §2.2 update.**
- **Phase 4 (optional) — coverage gate:** a marquee op dropping below its required
  bar fails CI (same shape as `drift_check.py`).

## Tests
`tests/test_coverage.py`: per-classifier unit tests on hand-built row sets (incl.
flame `scale_capped`+`n/a`, a registration `real`+`domain` op); a small golden
`coverage_deficit.json` snapshot. `ruff` + existing contract tests stay green;
drift unaffected (no op outputs change). The two refactors verified by
regenerating ECONOMIC.md / SCALING.md and confirming no diff.
