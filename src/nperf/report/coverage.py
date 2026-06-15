# -*- coding: utf-8 -*-
"""Coverage-&-deficit report (L5; COVERAGE_MANDATE §2.2).

Joins the nitrix op **catalogue** (``op_matrix.json``, read purely as the list
of ops -- capability stays nitrix's) with the perf-bench L4 store (by
``Case.op_qualname``) and emits, per op: coverage status, reference strength,
precision, and -- where a strong on-target GPU reference ran -- the
apples-to-apples GPU ratio.  Then two ranked lists for the nitrix agent:

  (a) **under-covered** ops, by priority (unmeasured / missing a platform / no
      strong on-target reference); and
  (b) **measured-but-lagging** ops, by severity (nitrix slower than its strong
      on-target reference on the deployment platform).

This is the inverse of the op_matrix feed: the feed pushes the numbers we
*have*; this surfaces the *gaps* and the on-target *deficits* -- "the numerics
most in need of improvement".

No *metric* arithmetic (DESIGN §1 / SCHEMA §G): every ratio is read from the
stored rows (computed + stored in L1); this layer only classifies, selects, and
ranks.  The one derived figure -- "nitrix ≈N× slower" in the markdown -- is the
reciprocal of a stored ratio, a presentation transform (the JSON exposes the
stored ratio itself).  A ``fast`` (--skip-slow) run is marked **provisional**
and never blesses an op (mandate §7); rows without a ``coverage_mode``
(pre-guard) count as full.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from . import economic
from .sizing import size_elems

# coverage status (which platforms have an ok nitrix row).
UNMEASURED, CPU_ONLY, GPU_ONLY, MULTIPLATFORM = (
    'unmeasured', 'cpu_only', 'gpu_only', 'multiplatform',
)
# reference strength (best non-nitrix reference that produced an ok row).
NO_REF, INTERNAL_ONLY, FLOOR_ONLY, STRONG_REF = (
    'none', 'internal_only', 'floor_only', 'strong_ref',
)
_RANK = {NO_REF: 0, INTERNAL_ONLY: 1, FLOOR_ONLY: 2, STRONG_REF: 3}

_GPU = 'jax-cuda12'
_CPU = 'jax-cpu'
_NITRIX = 'nitrix-jax'  # the system under test / ratio reference

# Community gold-standard domains (the real baselines for marquee functions):
# keyed on the baseline-NAME namespace, NOT framework -- the CLI tools (AFNI /
# FSL / FreeSurfer) are subprocess binaries declared ``framework='numpy'``, so
# a framework check would misclassify them as a host floor.  This is the fix
# for the latent deficit: these gold standards were invisible to ref_strength.
_DOMAIN_NS = frozenset({
    'ants', 'fsl', 'afni', 'freesurfer', 'dipy', 'simpleitk', 'statsmodels',
})

# Curated *community* baselines: named libraries optimised over years by expert
# engineers (scipy/sklearn + MONAI for medical imaging + the domain CLIs).  The
# CPU-CPU lens (below) races nitrix against the fastest of these ON CPU -- a
# second view of how close nitrix's *algorithm* is to optimal, independent of
# the GPU.  Our own ``numpy.*`` reimplementation-oracles and ``*.iofloor``
# no-ops are NOT community competitors and are deliberately excluded (a naive
# numpy reimpl would manufacture a flattering or meaningless gap).
_COMMUNITY_NS = frozenset({'scipy', 'sklearn', 'monai'}) | _DOMAIN_NS
# A community CPU baseline that outpaces nitrix on CPU by >= this factor is an
# independent "optimise the algorithm" signal -- it fires even when the GPU
# economic + performance bars are cleared (it SUPPLEMENTS, never supersedes,
# those critical signals).  ~1.5-2x is the user's bar; tunable (--cpu-gap).
CPU_OPTIMIZE_GAP = 1.5


def _is_community(baseline: str) -> bool:
    '''A curated community-library baseline (scipy/sklearn/MONAI/ANTs/...) --
    not our own numpy reimplementation-oracle, not an iofloor no-op.  Keyed on
    the name namespace (the domain CLIs are ``framework='numpy'``).'''
    if baseline.endswith('.iofloor'):
        return False
    return baseline.split('.')[0] in _COMMUNITY_NS


def _ref_class(baseline: str, framework: str) -> str:
    '''Classify a baseline: the system under test (nitrix), a community-gold
    DOMAIN reference (ANTs/FSL/...), a CPU floor (numpy/scipy or an I/O no-op),
    a strong on-target GPU reference (cupy/torch/pyg), or an internal jax
    alternative.  Domain is keyed on the name namespace (the CLI tools are
    ``framework='numpy'``); the ``*.iofloor`` no-ops are floors, not refs.'''
    if baseline.startswith('nitrix'):
        return 'nitrix'
    if baseline.endswith('.iofloor'):
        return 'floor'             # an I/O-timing no-op, not a reference
    if baseline.split('.')[0] in _DOMAIN_NS:
        return 'domain'            # community gold standard (ANTs/FSL/...)
    if framework == 'numpy':       # the numpy + scipy providers (host floor)
        return 'floor'
    if framework in ('cupy', 'torch'):
        return 'strong'            # external on-target ref (cupy/torch/pyg)
    return 'internal'              # jax, non-nitrix (naive-dense, jnp-matmul)


@dataclass(frozen=True)
class OpCoverage:
    '''Per-op coverage + deficit record (one catalogue op).'''
    qualname: str
    runtime: bool        # False = host-side constructor (jit n/a)
    has_case: bool       # is there a perf-bench case for this op?
    coverage: str        # unmeasured|cpu_only|gpu_only|multiplatform
    ref_strength: str    # none|internal_only|floor_only|strong_ref
    precision: str       # unmeasured|f32_only|multi_dtype
    provisional: bool    # latest rows from a fast (--skip-slow) run
    # nitrix's GPU attempt was skipped for an involuntary backend/solver reason
    # while a strong GPU ref ran -- GPU-capable but nitrix-blocked (e.g. the
    # jaxlib cuSOLVER blocker).  ``gpu_block_reason`` is the *recorded* reason.
    gpu_blocked: bool = False
    gpu_block_reason: Optional[str] = None
    # apples-to-apples GPU bar at the representative point, if a strong GPU ref
    # ran: the STORED ratio ``strong_ref.min / nitrix.min`` -- < 1 == nitrix
    # slower (a deficit).  None when no strong GPU ref was measured.
    gpu_ref: Optional[str] = None
    gpu_ref_ratio: Optional[float] = None
    # CPU-CPU community lens (supplements the GPU signals -- never supersedes):
    # the fastest curated community CPU baseline (scipy/sklearn/MONAI/ANTs/...)
    # vs nitrix ON CPU at the representative point, as the STORED ratio
    # ``community.min / nitrix_cpu.min`` (<1 => nitrix slower on CPU).  A large
    # gap is an independent "optimise this algorithm" candidate.  None when no
    # community CPU baseline ran ok there.
    cpu_ref: Optional[str] = None
    cpu_ref_ratio: Optional[float] = None
    # --- COVERAGE v2 axes (COVERAGE_V2_PLAN.md) ----------------------------
    # scale: did the op's declared brain-scale tier (large_param_points) run?
    # no_tier (none declared) / declared (declared, not measured on GPU) /
    # scaled (nitrix ok at the largest) / scale_capped (oom/timeout at a large
    # point while a smaller one ran -- the fragile-at-scale signal).
    scale_status: str = 'no_tier'
    largest_ok_size: Optional[int] = None
    scale_cap_reason: Optional[str] = None      # 'oom' | 'timeout'
    # economic: GPU as a multiple of the CPU gold standard (decision 4: the
    # largest real/large point, else the representative + authoritative=False).
    economic_verdict: str = 'n/a'               # see report/economic.verdict
    economic_amortized: Optional[float] = None
    economic_authoritative: bool = True
    # input realism (the achieved rung, max over ok nitrix rows): synthetic /
    # real_planted (real data, planted/known truth) / real_full (the actual
    # problem).
    input_realism: str = 'synthetic'
    # community gold-standard ref (ANTs/FSL/...) + the realism rung it ran
    # at (was it run on real data?).  None when no domain ref was measured.
    domain_ref: Optional[str] = None
    domain_ref_realism: str = 'synthetic'
    # benchmarking-policy tier (from the Case): 'marquee' ops must reach real
    # data + a domain ref on real data; 'standard' are not held to that bar.
    tier: str = 'standard'

    @property
    def nitrix_slower_on_gpu(self) -> bool:
        return self.gpu_ref_ratio is not None and self.gpu_ref_ratio < 1.0

    @property
    def cpu_gap(self) -> Optional[float]:
        '''How many x faster the fastest community CPU baseline is than nitrix
        on CPU (``1 / cpu_ref_ratio``); >1 => nitrix slower.  None when no
        community CPU baseline ran.'''
        if not self.cpu_ref_ratio or self.cpu_ref_ratio <= 0:
            return None
        return 1.0 / self.cpu_ref_ratio


def _same_point(pp: Dict[str, Any], rep: Dict[str, Any]) -> bool:
    '''A row's param point matches the representative (every key but seed).'''
    return all(pp.get(k) == v for k, v in rep.items() if k != 'seed')


def _coverage_status(rows: List[Dict[str, Any]]) -> str:
    plats = {
        r['platform'] for r in rows
        if r.get('baseline') == _NITRIX and r.get('status') == 'ok'
    }
    cpu, gpu = _CPU in plats, _GPU in plats
    if cpu and gpu:
        return MULTIPLATFORM
    if gpu:
        return GPU_ONLY
    if cpu:
        return CPU_ONLY
    return UNMEASURED


# nitrix GPU attempt skipped for an *involuntary* backend/solver reason (not a
# config/slow opt-out).  Reason-robust: a set, so a future failure mode is
# easy to add and the report shows the *recorded* reason, never an assumed one.
_BLOCKED_REASONS = frozenset({'gpu_solver_unavailable', 'backend_unavailable'})


def _gpu_block_reason(rows: List[Dict[str, Any]]) -> Optional[str]:
    '''If nitrix's GPU attempt was skipped for an involuntary backend/solver
    reason AND a strong external GPU ref ran ok on GPU, return that recorded
    reason -- the GPU is provably capable of the op but nitrix's path can't use
    it (e.g. the jaxlib cuSOLVER blocker for the eigh family).  Else ``None``.

    Requiring a working strong GPU ref is what distinguishes "blocked but
    GPU-capable" from "nitrix can't GPU this and nor can anything" -- and the
    reason is read from the row, never assumed, so it stays honest if the
    surfaced failure changes.'''
    reason = None
    for r in rows:
        if (r.get('platform') == _GPU and r.get('baseline') == _NITRIX
                and r.get('status') == 'skipped'):
            rs = (r.get('failure_detail') or {}).get('reason')
            if rs in _BLOCKED_REASONS:
                reason = rs
                break
    if reason is None:
        return None
    strong_ran = any(
        r.get('platform') == _GPU and r.get('status') == 'ok'
        and _ref_class(r.get('baseline', ''),
                       r.get('framework', '')) == 'strong'
        for r in rows
    )
    return reason if strong_ran else None


def _ref_strength(rows: List[Dict[str, Any]]) -> str:
    best = NO_REF
    for r in rows:
        if r.get('status') != 'ok':
            continue
        cls = _ref_class(r.get('baseline', ''), r.get('framework', ''))
        cand = {'floor': FLOOR_ONLY, 'strong': STRONG_REF,
                'internal': INTERNAL_ONLY}.get(cls)
        if cand and _RANK[cand] > _RANK[best]:
            best = cand
    return best


def _precision(rows: List[Dict[str, Any]]) -> str:
    dtypes = {
        (r.get('param_point') or {}).get('dtype', 'f32')
        for r in rows if r.get('status') == 'ok'
    }
    if not dtypes:
        return 'unmeasured'
    return 'multi_dtype' if len(dtypes) > 1 else 'f32_only'


def _gpu_bar(
    rows: List[Dict[str, Any]], rep: Dict[str, Any]
) -> Tuple[Optional[str], Optional[float]]:
    '''The strong on-target GPU ref + its stored ratio at the representative
    point (the apples-to-apples bar), or (None, None) if none ran ok there.'''
    for r in rows:
        if (r.get('platform') == _GPU and r.get('status') == 'ok'
                and _ref_class(r.get('baseline', ''),
                               r.get('framework', '')) == 'strong'
                and r.get('ratio') and _same_point(r.get('param_point', {}),
                                                    rep)):
            return r['baseline'], r['ratio'].get('value')
    return None, None


def _cpu_bar(
    rows: List[Dict[str, Any]], rep: Dict[str, Any]
) -> Tuple[Optional[str], Optional[float]]:
    '''The *fastest* curated community CPU baseline + its stored ratio at the
    representative point on jax-cpu (``community.min / nitrix_cpu.min``; <1 =>
    nitrix slower on CPU).  The fastest community tool (smallest ratio) is the
    strongest evidence of room to optimise, so it is the one reported.  (None,
    None) if no community baseline ran ok on CPU there.'''
    best_name, best_ratio = None, None
    for r in rows:
        if (r.get('platform') == _CPU and r.get('status') == 'ok'
                and _is_community(r.get('baseline', ''))
                and r.get('ratio')
                and _same_point(r.get('param_point', {}), rep)):
            v = r['ratio'].get('value')
            if v is not None and (best_ratio is None or v < best_ratio):
                best_name, best_ratio = r['baseline'], v
    return best_name, best_ratio


def _lags_on_cpu(oc: 'OpCoverage', gap: float = CPU_OPTIMIZE_GAP) -> bool:
    '''nitrix is >= ``gap`` x slower than the community CPU baseline.'''
    g = oc.cpu_gap
    return g is not None and g >= gap


# scale: statuses (COVERAGE_V2_PLAN) + cap reasons marking a fragile tier.
NO_TIER, SCALE_DECLARED, SCALED, SCALE_CAPPED = (
    'no_tier', 'declared', 'scaled', 'scale_capped')
_CAP_STATUSES = frozenset({'oom', 'timeout'})


def _scale_status(
    case: Any, crows: List[Dict[str, Any]]
) -> Tuple[str, Optional[int], Optional[str]]:
    '''Did the op's declared brain-scale tier actually run?  Platform-agnostic
    (a GPU-blocked op like flame still has a CPU scale story): scaled if a
    nitrix ok row reaches the largest declared size; scale_capped if nitrix
    oom/timeouts at a large point beyond the largest size that ran (the
    fragile-at-scale signal); declared if the tier exists but isn't measured;
    no_tier if none declared.'''
    large = list(case.large_param_points)
    if not large:
        return NO_TIER, None, None
    keys = {economic._pkey(p) for p in large}
    nrows = [r for r in crows if r.get('baseline') == _NITRIX
             and economic._pkey(r['param_point']) in keys]
    ok = [r for r in nrows if r.get('status') == 'ok']
    capped = [r for r in nrows if r.get('status') in _CAP_STATUSES]
    if not ok and not capped:
        return SCALE_DECLARED, None, None
    largest_ok = max((size_elems(r['param_point']) for r in ok), default=None)
    above = [r for r in capped if largest_ok is None
             or size_elems(r['param_point']) > largest_ok]
    if above:
        worst = max(above, key=lambda r: size_elems(r['param_point']))
        return SCALE_CAPPED, largest_ok, worst['status']
    largest_declared = max(size_elems(p) for p in large)
    if largest_ok is not None and largest_ok >= largest_declared:
        return SCALED, largest_ok, None
    return SCALE_DECLARED, largest_ok, None


def _input_realism(crows: List[Dict[str, Any]]) -> str:
    '''The highest realism rung reached by an ok nitrix row (synthetic <
    real_planted < real_full).'''
    rung = 'synthetic'
    for r in crows:
        if r.get('baseline') == _NITRIX and r.get('status') == 'ok':
            rr = economic.realism_rung(r['param_point'])
            if economic.rung_index(rr) > economic.rung_index(rung):
                rung = rr
    return rung


def _domain_ref(crows: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    '''The community gold-standard reference that ran ok, preferring the one
    measured on the most-real data; returns (baseline, its realism rung).'''
    best: Optional[str] = None
    rung = 'synthetic'
    for r in crows:
        cls = _ref_class(r.get('baseline', ''), r.get('framework', ''))
        if r.get('status') != 'ok' or cls != 'domain':
            continue
        rr = economic.realism_rung(r['param_point'])
        if best is None or economic.rung_index(rr) > economic.rung_index(rung):
            best, rung = r['baseline'], rr
    return best, rung


def _economic(case: Any, crows: List[Dict[str, Any]], coverage: str,
              bar: float) -> Tuple[str, Optional[float], bool]:
    '''The economic verdict for the op (decision 4).  ``n/a`` when nitrix does
    not run on GPU at all (blocked / cpu-only) -- there is no GPU win to weigh;
    else reduce the shared economic join to one verdict.'''
    if coverage in (UNMEASURED, CPU_ONLY):
        return 'n/a', None, True
    ov = economic.op_verdict(case, crows, bar)
    return ov.verdict, ov.amortized, ov.authoritative


def build_coverage(
    rows: List[Dict[str, Any]],
    catalogue: List[Dict[str, Any]],
    op_to_case: Dict[str, Any],
    bar: float = economic.COST_MULTIPLE,
) -> List[OpCoverage]:
    '''One ``OpCoverage`` per catalogue op, joining the store ``rows`` by case.

    ``catalogue`` is ``op_matrix.json``'s ``ops`` list (the authoritative op
    list); ``op_to_case`` maps an op qualname to its ``Case`` (from the case
    registry).  An op with no case -- or a case with no rows -- is
    ``unmeasured``.  ``bar`` is the GPU:CPU cost-multiple for the economic
    axis.'''
    by_case: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r.get('case'), []).append(r)
    out: List[OpCoverage] = []
    for op in catalogue:
        q = op.get('qualname')
        runtime = op.get('jit') != 'n/a'
        case = op_to_case.get(q)
        crows = by_case.get(case.name, []) if case else []
        if not case or not crows:
            out.append(OpCoverage(
                qualname=q, runtime=runtime, has_case=bool(case),
                coverage=UNMEASURED, ref_strength=NO_REF,
                precision='unmeasured', provisional=False,
                tier=(getattr(case, 'tier', 'standard')
                      if case else 'standard')))
            continue
        coverage = _coverage_status(crows)
        gpu_ref, gpu_ratio = _gpu_bar(crows, case.representative)
        cpu_ref, cpu_ratio = _cpu_bar(crows, case.representative)
        block_reason = _gpu_block_reason(crows)
        scale, largest_ok, cap_reason = _scale_status(case, crows)
        econ_v, econ_amort, econ_auth = _economic(case, crows, coverage, bar)
        dref, dref_rung = _domain_ref(crows)
        out.append(OpCoverage(
            qualname=q, runtime=runtime, has_case=True,
            coverage=coverage,
            ref_strength=_ref_strength(crows),
            precision=_precision(crows),
            provisional=any(
                (r.get('provenance') or {}).get('coverage_mode') == 'fast'
                for r in crows),
            gpu_blocked=block_reason is not None,
            gpu_block_reason=block_reason,
            gpu_ref=gpu_ref, gpu_ref_ratio=gpu_ratio,
            cpu_ref=cpu_ref, cpu_ref_ratio=cpu_ratio,
            scale_status=scale, largest_ok_size=largest_ok,
            scale_cap_reason=cap_reason,
            economic_verdict=econ_v, economic_amortized=econ_amort,
            economic_authoritative=econ_auth,
            input_realism=_input_realism(crows),
            domain_ref=dref, domain_ref_realism=dref_rung,
            tier=getattr(case, 'tier', 'standard')))
    return out


def _priority(oc: OpCoverage) -> Optional[str]:
    '''Priority tier for the under-covered list, or ``None`` if the op is fully
    covered (multiplatform + a strong on-target ref) or a host-side constructor
    (tracked separately).  A coarse, transparent heuristic -- consumer-traffic
    weighting is a future input (mandate §2.2/§4).'''
    if not oc.runtime:
        return None
    if oc.gpu_blocked:
        return None                    # tracked in the GPU-blocked section
    if oc.coverage in (UNMEASURED, CPU_ONLY, GPU_ONLY):
        return 'high'                  # no data, or missing a platform
    if oc.ref_strength != STRONG_REF:
        return 'medium'                # measured both, but no GPU bar
    return None                        # multiplatform + strong ref: covered


def _op_json(r: OpCoverage) -> Dict[str, Any]:
    return {
        'qualname': r.qualname, 'runtime': r.runtime, 'has_case': r.has_case,
        'coverage': r.coverage, 'ref_strength': r.ref_strength,
        'precision': r.precision, 'provisional': r.provisional,
        'gpu_blocked': r.gpu_blocked, 'gpu_block_reason': r.gpu_block_reason,
        'gpu_ref': r.gpu_ref, 'gpu_ref_ratio': r.gpu_ref_ratio,
        'nitrix_slower_on_gpu': r.nitrix_slower_on_gpu,
        'cpu_ref': r.cpu_ref, 'cpu_ref_ratio': r.cpu_ref_ratio,
        'cpu_gap': r.cpu_gap, 'nitrix_lagging_on_cpu': _lags_on_cpu(r),
        'scale_status': r.scale_status, 'largest_ok_size': r.largest_ok_size,
        'scale_cap_reason': r.scale_cap_reason,
        'economic_verdict': r.economic_verdict,
        'economic_amortized': r.economic_amortized,
        'economic_authoritative': r.economic_authoritative,
        'input_realism': r.input_realism,
        'domain_ref': r.domain_ref, 'domain_ref_realism': r.domain_ref_realism,
        'tier': r.tier, 'coverage_score': list(score(r)),
    }


def _lagging(records: List[OpCoverage]) -> List[OpCoverage]:
    return sorted((r for r in records if r.nitrix_slower_on_gpu),
                  key=lambda r: r.gpu_ref_ratio or 0.0)


def _cpu_lagging(records: List[OpCoverage],
                 gap: float = CPU_OPTIMIZE_GAP) -> List[OpCoverage]:
    '''Measured ops where a curated community CPU baseline outpaces nitrix on
    CPU by >= ``gap`` x -- the algorithm-optimise candidates (worst gap first).
    Independent of the GPU verdict: a CPU-lagging op can still be a GPU win;
    this is a supplementary lens, not a gate.'''
    return sorted((r for r in records if _lags_on_cpu(r, gap)),
                  key=lambda r: r.cpu_ref_ratio or 1.0)


def _under(records: List[OpCoverage]) -> List[Tuple[OpCoverage, str]]:
    pairs = [(r, _priority(r)) for r in records if _priority(r)]
    return sorted(pairs, key=lambda rp: (
        {'high': 0, 'medium': 1}[rp[1]], rp[0].qualname))


# --- COVERAGE v2 per-axis deficit selectors -------------------------------
def _scale_fragile(records: List[OpCoverage]) -> List[OpCoverage]:
    '''Declared a brain-scale tier but nitrix oom/timeouts before the top.'''
    return sorted((r for r in records if r.scale_status == SCALE_CAPPED),
                  key=lambda r: r.largest_ok_size or 0)


def _no_econ_win(records: List[OpCoverage]) -> List[OpCoverage]:
    '''A real GPU op whose win is below the cost-multiple bar.'''
    return sorted(
        (r for r in records
         if r.economic_verdict == 'not multiplicative enough'),
        key=lambda r: r.economic_amortized or 0.0)


def _real_data(records: List[OpCoverage]) -> List[OpCoverage]:
    '''Ops measured on real data (planted or full), most-real first.'''
    return sorted(
        (r for r in records if r.input_realism != 'synthetic'),
        key=lambda r: -economic.rung_index(r.input_realism))


# --- COVERAGE v2 tier-gated score (Phase 2) -------------------------------
def axes_status(oc: OpCoverage) -> List[Tuple[str, bool]]:
    '''The applicable required-*coverage* axes for ``oc``'s tier, each
    (name, satisfied?).  General axes (platform / reference / scale-if-tiered)
    apply to every measured op; marquee ops also require real-data input + a
    domain ref measured on real data (the headline bar).  The economic verdict
    is a result/indicator, not scored here (see below).'''
    rp = economic.rung_index('real_planted')
    axes: List[Tuple[str, bool]] = [
        ('platform', oc.coverage == MULTIPLATFORM or oc.gpu_blocked),
        ('reference', oc.ref_strength == STRONG_REF
         or oc.domain_ref is not None),
    ]
    if oc.scale_status != NO_TIER:
        axes.append(('scale', oc.scale_status == SCALED))
    # NB: the economic verdict is an *indicator/result* (is the op a GPU win),
    # not a coverage requirement -- it is a first-class matrix column + its own
    # deficit section, but is deliberately NOT in the completeness score (a
    # not-multiplicative op is a finding, not a coverage gap).
    if oc.tier == 'marquee':
        axes.append(
            ('real_input', economic.rung_index(oc.input_realism) >= rp))
        axes.append(
            ('domain_on_real', oc.domain_ref is not None
             and economic.rung_index(oc.domain_ref_realism) >= rp))
    return axes


def score(oc: OpCoverage) -> Tuple[int, int]:
    '''(satisfied, applicable) over the tier's required axes.'''
    a = axes_status(oc)
    return sum(1 for _, ok in a if ok), len(a)


def _marquee_unmet(records: List[OpCoverage]) -> List[OpCoverage]:
    '''Marquee ops not yet on real data, or lacking a domain ref on real data
    -- the distinctive marquee bar (worst score first).'''
    out = []
    for r in records:
        if r.tier != 'marquee':
            continue
        rp = economic.rung_index('real_planted')
        real_ok = economic.rung_index(r.input_realism) >= rp
        dom_ok = (r.domain_ref is not None
                  and economic.rung_index(r.domain_ref_realism) >= rp)
        if not (real_ok and dom_ok):
            out.append(r)
    return sorted(out, key=lambda r: score(r)[0] - score(r)[1])


def render_json(records: List[OpCoverage],
                orphans: List[str] = (),
                cpu_gap: float = CPU_OPTIMIZE_GAP) -> Dict[str, Any]:
    '''Machine-readable artifact for the nitrix agent (the ranked deficits).
    ``orphans`` = ``(qualname, tier)`` for ops that HAVE a perf-bench case (and
    are benchmarked) but are absent from the nitrix catalogue -- invisible to
    the join until the catalogue is regenerated.  ``cpu_gap`` is the CPU-CPU
    community-baseline factor at which an op is flagged as an optimise
    candidate.'''
    runtime = [r for r in records if r.runtime]

    def _n(pred: Any) -> int:
        return sum(1 for r in runtime if pred(r))

    lagging, under = _lagging(records), _under(records)
    cpu_lag = _cpu_lagging(records, cpu_gap)
    blocked = [r for r in records if r.gpu_blocked]
    fragile, no_win = _scale_fragile(records), _no_econ_win(records)
    real = _real_data(records)
    marquee = [r for r in records if r.tier == 'marquee']
    unmet = _marquee_unmet(records)
    return {
        'source': 'nitrix-perf-bench coverage-&-deficit report',
        'convention': 'gpu_ref_ratio = strong_ref.min / nitrix.min '
                      '(<1 => nitrix slower on the GPU)',
        'summary': {
            'runtime_ops': len(runtime),
            'measured': _n(lambda r: r.coverage != UNMEASURED),
            'multiplatform': _n(lambda r: r.coverage == MULTIPLATFORM),
            'with_strong_gpu_ref': _n(lambda r: r.ref_strength == STRONG_REF),
            'with_domain_ref': _n(lambda r: r.domain_ref is not None),
            'lagging_on_gpu': len(lagging),
            'lagging_on_cpu_vs_community': len(cpu_lag),
            'gpu_blocked_upstream': len(blocked),
            'constructors': sum(1 for r in records if not r.runtime),
            # COVERAGE v2 axes
            'scaled': _n(lambda r: r.scale_status == SCALED),
            'scale_capped': len(fragile),
            'economic_favorable': _n(
                lambda r: r.economic_verdict.startswith('favorable')),
            'economic_not_multiplicative': len(no_win),
            'on_real_data': len(real),
            'on_real_full': _n(lambda r: r.input_realism == 'real_full'),
            'marquee': len(marquee),
            'marquee_unmet': len(unmet),
            'orphan_cases': len(orphans),
        },
        'orphan_cases': [{'qualname': q, 'tier': t} for q, t in orphans],
        'lagging': [_op_json(r) for r in lagging],
        'cpu_lagging_vs_community': [_op_json(r) for r in cpu_lag],
        'gpu_blocked': [_op_json(r) for r in blocked],
        'under_covered': [{**_op_json(r), 'priority': p} for r, p in under],
        'by_axis': {
            'scale_fragile': [_op_json(r) for r in fragile],
            'no_economic_win': [_op_json(r) for r in no_win],
            'on_real_data': [_op_json(r) for r in real],
            'marquee_unmet': [_op_json(r) for r in unmet],
        },
        'ops': [_op_json(r) for r in records],
    }


def _slower(ratio: Optional[float]) -> str:
    '''Human "nitrix ≈N× slower/faster" from the stored ratio -- a presentation
    transform of an L1 value (the JSON exposes the ratio itself; SCHEMA §G).'''
    if ratio is None:
        return '-'
    if ratio < 1:
        return f'~{1.0 / ratio:.1f}x slower'
    return f'~{ratio:.1f}x faster'


_MATRIX_HEAD = ('platform', 'scale', 'economic', 'input', 'gpu-ref',
                'domain-ref')


def _matrix_cells(r: OpCoverage) -> List[str]:
    '''The glyph cells for one op's coverage-matrix row: ``[op, score,
    platform, scale, economic, input, gpu-ref, domain-ref]``.  Shared by the
    marquee matrix and the full matrix.'''
    sat, app = score(r)
    plat = ('✓' if r.coverage == MULTIPLATFORM
            else '⊘blk' if r.gpu_blocked else f'✗ {r.coverage}')
    scl = {NO_TIER: '·', SCALED: '✓', SCALE_DECLARED: '○',
           SCALE_CAPPED: f'⚠ {r.scale_cap_reason}'}[r.scale_status]
    if r.economic_verdict == 'n/a':
        eco = '·'
    else:
        g = '✓' if r.economic_verdict.startswith('favorable') else '✗'
        eco = f'{g}{"" if r.economic_authoritative else "~"}'
    inp = {'synthetic': '✗ synth', 'real_planted': '◐ planted',
           'real_full': '● full'}[r.input_realism]
    gref = '✓' if r.ref_strength == STRONG_REF else '·'
    if r.domain_ref is None:
        dref = '✗ none'
    else:
        on = (economic.rung_index(r.domain_ref_realism)
              >= economic.rung_index('real_planted'))
        dref = f'{"●" if on else "◐"} {r.domain_ref}'
    return [f'`{r.qualname}`', f'{sat}/{app}', plat, scl, eco, inp, gref, dref]


def render_markdown(records: List[OpCoverage],
                    orphans: List[str] = (),
                    cpu_gap: float = CPU_OPTIMIZE_GAP) -> str:
    '''Human-facing report: the ranked lists + the coverage matrix + summary.
    ``orphans`` = benchmarked cases absent from the nitrix catalogue.
    ``cpu_gap`` is the community-baseline factor flagging a CPU-optimise op.'''
    runtime = [r for r in records if r.runtime]
    meas = sum(1 for r in runtime if r.coverage != UNMEASURED)
    multi = sum(1 for r in runtime if r.coverage == MULTIPLATFORM)
    strong = sum(1 for r in runtime if r.ref_strength == STRONG_REF)
    domain = sum(1 for r in runtime if r.domain_ref is not None)
    scaled = sum(1 for r in runtime if r.scale_status == SCALED)
    econ_fav = sum(1 for r in runtime
                   if r.economic_verdict.startswith('favorable'))
    on_real = sum(1 for r in runtime if r.input_realism != 'synthetic')
    lagging, under = _lagging(records), _under(records)
    cpu_lag = _cpu_lagging(records, cpu_gap)
    blocked = [r for r in records if r.gpu_blocked]
    fragile, no_win, real = (
        _scale_fragile(records), _no_econ_win(records), _real_data(records))
    marquee = sorted((r for r in records if r.tier == 'marquee'),
                     key=lambda r: (score(r)[0] - score(r)[1], r.qualname))
    unmet = _marquee_unmet(records)
    won = sorted(
        (r for r in records if r.ref_strength == STRONG_REF
         and r.gpu_ref_ratio is not None and not r.nitrix_slower_on_gpu),
        key=lambda r: -(r.gpu_ref_ratio or 0.0))

    lines: List[str] = [
        '# nitrix-perf-bench — coverage & deficit report',
        '',
        '> Generated from the L4 store joined with the nitrix op catalogue '
        '(`op_matrix.json`). No values are hand-edited; every ratio is read '
        'from the stored rows (SCHEMA §G).',
        '',
        '## Coverage (runtime ops)',
        '',
        f'- **runtime ops catalogued**: {len(runtime)} '
        f'(+ {len(records) - len(runtime)} host-side constructors, apart)',
        f'- **measured** (≥1 platform): {meas} / {len(runtime)}',
        f'- **multiplatform** (CPU + GPU): {multi} / {len(runtime)}',
        f'- **with a strong on-target GPU ref**: {strong} / {len(runtime)}',
        f'- **with a community-gold ref** (ANTs/FSL/…): {domain} '
        f'/ {len(runtime)}',
        f'- **scaled** (ran at the declared brain-scale tier): {scaled} '
        f'/ {len(runtime)} — **{len(fragile)}** fragile (oom/timeout)',
        f'- **economically favorable** (GPU beats CPU gold by ≥ the bar): '
        f'{econ_fav} / {len(runtime)} — **{len(no_win)}** not multiplicative',
        f'- **on real data** (planted or full): {on_real} / {len(runtime)}',
        f'- **marquee** ops (held to the real-data + community-baseline bar): '
        f'{len(marquee)} — **{len(unmet)}** not yet meeting it',
        f'- **lagging on the GPU**: {len(lagging)}',
        f'- **lagging on CPU vs the community baseline** (≥{cpu_gap:g}×, '
        f'an optimise signal): {len(cpu_lag)}',
        f'- **GPU blocked upstream** (jaxlib cuSOLVER): {len(blocked)}',
    ]
    if orphans:
        marq = [q for q, t in orphans if t == 'marquee']
        note = (
            f'- ⚠️ **{len(orphans)} benchmarked case(s) absent from the '
            'catalogue** (`op_matrix.json` is stale -- invisible to the join '
            'until regenerated in nitrix): '
            + ', '.join(f'`{q.split(".")[-1]}`' for q, _ in orphans) + '.')
        if marq:
            note += (' Includes **MARQUEE** ops: '
                     + ', '.join(f'`{q.split(".")[-1]}`' for q in marq) + '.')
        lines.append(note)
    lines += [
        '',
        '## Lagging on the deployment target (GPU) — ranked',
        '',
    ]
    if lagging:
        lines += [
            'nitrix is slower than its strong on-target reference here '
            '(`ratio = ref/nitrix < 1`); worst first. The Pallas-kernel / '
            'algorithm candidates.',
            '',
            '| # | op | strong GPU ref | ratio (ref/nitrix) | nitrix | note |',
            '|---|---|---|---:|---|---|',
        ]
        for i, r in enumerate(lagging, 1):
            note = 'provisional (fast run)' if r.provisional else ''
            lines.append(
                f'| {i} | `{r.qualname}` | {r.gpu_ref} | '
                f'{r.gpu_ref_ratio:.3g} | {_slower(r.gpu_ref_ratio)} | '
                f'{note} |')
    else:
        lines.append('_No op measured against a strong on-target GPU ref '
                     'yet._')
    lines += [
        '',
        '## Lagging on CPU vs the community baseline — ranked',
        '',
        'A **supplementary** lens (it does **not** supersede the strong-GPU '
        'and GPU-economic signals): nitrix-CPU vs the fastest curated '
        '*community* CPU baseline (scipy / sklearn / MONAI / ANTs / FSL / …, '
        'on `jax-cpu`), at the representative point. These libraries are '
        'optimised over years by expert engineers, so a large CPU gap is a '
        f'second read on how close nitrix\'s **algorithm** is to optimal — '
        f'≥{cpu_gap:g}× independently signals "optimise this", even when the '
        'op already clears the GPU economic + performance bars. (Our own '
        '`numpy.*` reimpl-oracles and `*.iofloor` no-ops are excluded — only '
        'named community libraries count.)',
        '',
    ]
    if cpu_lag:
        lines += [
            '| # | op | community CPU ref | gap (ref/nitrix) | nitrix |',
            '|---|---|---|---:|---|',
        ]
        for i, r in enumerate(cpu_lag, 1):
            note = ' · provisional' if r.provisional else ''
            lines.append(
                f'| {i} | `{r.qualname}` | {r.cpu_ref} | '
                f'{r.cpu_ref_ratio:.3g} | {_slower(r.cpu_ref_ratio)}{note} |')
    else:
        lines.append(
            f'_No op lags a community CPU baseline by ≥{cpu_gap:g}× '
            '(or none measured against one yet)._')
    if blocked:
        lines += [
            '',
            '## GPU blocked — nitrix path skipped, a GPU ref works',
            '',
            "nitrix's GPU attempt was skipped for the recorded reason below, "
            'while a strong external GPU ref **did** run ok on the GPU -- so '
            'the GPU is capable of the op but nitrix\'s path is not using it. '
            'For the eigh family the cause is the jaxlib cuSOLVER bug '
            '(jax-ml/jax #29042; CuPy works on identical wheels); these are '
            'benchmarked on CPU and the fix is upstream.',
            '',
            '| op | nitrix on GPU (skipped) |',
            '|---|---|',
        ]
        for r in blocked:
            lines.append(f'| `{r.qualname}` | {r.gpu_block_reason} |')
    lines += [
        '',
        '## Under-covered — ranked by priority',
        '',
        'Priority is a coarse heuristic (no consumer-traffic weighting yet): '
        '**high** = unmeasured or missing a platform; **medium** = measured '
        'on both but no strong on-target GPU ref (no apples-to-apples bar).',
        '',
        '| priority | op | coverage | ref strength | precision |',
        '|---|---|---|---|---|',
    ]
    for r, p in under:
        lines.append(
            f'| {p} | `{r.qualname}` | {r.coverage} | {r.ref_strength} | '
            f'{r.precision} |')
    if not under:
        lines.append('| - | _all runtime ops covered_ | | | |')
    if won:
        lines += [
            '',
            '## Covered with a strong GPU ref — nitrix ahead',
            '',
            '| op | strong GPU ref | ratio (ref/nitrix) | nitrix |',
            '|---|---|---:|---|',
        ]
        lines += [
            f'| `{r.qualname}` | {r.gpu_ref} | {r.gpu_ref_ratio:.3g} | '
            f'{_slower(r.gpu_ref_ratio)} |'
            for r in won
        ]
    # --- COVERAGE v2 axis sections ----------------------------------------
    lines += [
        '',
        '## Scale — brain-scale tier (COVERAGE v2)',
        '',
        'Ops declaring a `large_param_points` tier: did nitrix run at the '
        'largest realistic size, or break (oom/timeout) before it? '
        '`scale_capped` = **fragility at scale** (the win at a small '
        'size may not hold where practitioners run).',
        '',
    ]
    if fragile:
        lines += ['| op | scaled to | capped by |', '|---|---|---|']
        lines += [
            f'| `{r.qualname}` | {r.largest_ok_size or "—"} elem | '
            f'**{r.scale_cap_reason}** |' for r in fragile]
    else:
        lines.append('_No op is fragile at its declared scale tier._')
    lines += [
        '',
        '## Economic — GPU as a multiple of CPU (COVERAGE v2)',
        '',
        'The deployment-economics bar: a nitrix-GPU win counts only if it is '
        '**multiplicative** over the CPU gold standard (the GPU hardware '
        'premium; see `ECONOMIC.md`). Verdict at largest real/large point, '
        'else representative (`~` = not authoritative). `not multiplicative '
        'enough` is a real GPU win that still fails the cost test.',
        '',
        '| op | verdict | amortized | domain ref |',
        '|---|---|---:|---|',
    ]
    _na = ('n/a', 'unmeasured')
    econ_rows = sorted(
        (r for r in runtime if r.economic_verdict not in _na),
        key=lambda r: (r.economic_verdict != 'not multiplicative enough',
                       -(r.economic_amortized or 0.0)))
    if econ_rows:
        for r in econ_rows:
            amort = (f'{r.economic_amortized:.1f}x'
                     if r.economic_amortized is not None else '—')
            mark = '' if r.economic_authoritative else ' ~'
            lines.append(
                f'| `{r.qualname}` | {r.economic_verdict}{mark} | {amort} | '
                f'{r.domain_ref or "—"} |')
    else:
        lines.append('| _no GPU-vs-CPU-gold join yet_ | | | |')
    lines += [
        '',
        '## Real-data coverage (COVERAGE v2)',
        '',
        'Marquee functions should be tested on real brain data against real '
        'community baselines. Realism ladder: `synthetic` < `real_planted` '
        '(real image, planted/known truth) < `real_full` (actual problem). '
        'Which ops are *required* to reach real data is tier-gated -- see the '
        'marquee matrix below.',
        '',
    ]
    if real:
        lines += ['| op | realism | domain ref (on) |', '|---|---|---|']
        lines += [
            f'| `{r.qualname}` | {r.input_realism} | '
            f'{(r.domain_ref or "—")} ({r.domain_ref_realism}) |'
            for r in real]
    else:
        lines.append('_No op is yet measured on real data._')
    # --- the marquee coverage matrix (the first-class tier-gated view) ------
    lines += [
        '',
        '## Marquee coverage matrix (COVERAGE v2)',
        '',
        'The headline functions used on real images, scored against their '
        'tier bar (`score` = satisfied / applicable required axes). Glyphs: '
        '`✓` met · `✗` unmet · `⚠` fragile · `~` non-authoritative · '
        '`·` n/a. Worst-covered first.',
        '',
        '| op | score | platform | scale | economic | input | gpu-ref '
        '| domain-ref |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in marquee:
        lines.append('| ' + ' | '.join(_matrix_cells(r)) + ' |')
    if unmet:
        lines += [
            '',
            '**Marquee unmet** (no real-data input, or no domain ref on real '
            'data) — the next-round targets: '
            + ', '.join(f'`{r.qualname.split(".")[-1]}`' for r in unmet) + '.']
    # --- the full coverage matrix: every measured op, worst-vs-tier first ---
    measured = sorted(
        (r for r in records if r.has_case and r.runtime),
        key=lambda r: (score(r)[0] - score(r)[1], r.tier != 'marquee',
                       r.qualname))
    lines += [
        '',
        '## Full coverage matrix — every op with a case (COVERAGE v2)',
        '',
        f'All {len(measured)} ops with a case, scored against their tier '
        '(`★` = marquee, which adds the real-data + domain-on-real bar). '
        'Worst-covered (and marquee) first; same glyphs as above.',
        '',
        '| op | ★ | score | ' + ' | '.join(_MATRIX_HEAD) + ' |',
        '|---|---|---|' + '---|' * len(_MATRIX_HEAD),
    ]
    for r in measured:
        c = _matrix_cells(r)
        star = '★' if r.tier == 'marquee' else ''
        lines.append(f'| {c[0]} | {star} | ' + ' | '.join(c[1:]) + ' |')
    lines += [
        '',
        '## Caveats',
        '',
        "- `ratio = strong_ref.min / nitrix.min` at the op's representative "
        'point; `<1` ⇒ nitrix slower. The "≈Nx" column is its reciprocal '
        '(presentation only).',
        '- A **provisional** op\'s latest data came from a `--skip-slow` '
        '(fast) run; run the full sweep before acting (mandate §7).',
        '- "Lagging" is currently *slower than the strong on-target ref*; '
        'per-op **targets** (mandate §2.4) will refine the bar.',
        '- The **CPU-vs-community** gap (`community.min / nitrix_cpu.min` at '
        'the representative point, fastest community tool) is a supplementary '
        'algorithm-quality signal; it never supersedes the GPU economic / '
        'performance verdicts, and excludes our own numpy reimpl-oracles.',
        '- Host-side constructors (jit `n/a`) are excluded from the runtime '
        'denominator; they have no device-time bar.',
        '',
    ]
    return '\n'.join(lines) + '\n'

