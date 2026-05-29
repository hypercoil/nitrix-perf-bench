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


def _ref_class(baseline: str, framework: str) -> str:
    '''Classify a baseline by its row's ``framework``: the system under test,
    a same-framework internal alternative, a CPU floor (numpy/scipy), or a
    strong on-target external reference (cupy / torch / pyg).'''
    if baseline.startswith('nitrix'):
        return 'nitrix'
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
    # apples-to-apples GPU bar at the representative point, if a strong GPU ref
    # ran: the STORED ratio ``strong_ref.min / nitrix.min`` -- < 1 == nitrix
    # slower (a deficit).  None when no strong GPU ref was measured.
    gpu_ref: Optional[str] = None
    gpu_ref_ratio: Optional[float] = None

    @property
    def nitrix_slower_on_gpu(self) -> bool:
        return self.gpu_ref_ratio is not None and self.gpu_ref_ratio < 1.0


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


def build_coverage(
    rows: List[Dict[str, Any]],
    catalogue: List[Dict[str, Any]],
    op_to_case: Dict[str, Tuple[str, Dict[str, Any]]],
) -> List[OpCoverage]:
    '''One ``OpCoverage`` per catalogue op, joining the store ``rows`` by case.

    ``catalogue`` is ``op_matrix.json``'s ``ops`` list (the authoritative op
    list); ``op_to_case`` maps an op qualname to its ``(case_name,
    representative)`` (from the case registry).  An op with no case -- or a
    case with no rows -- is ``unmeasured``.'''
    by_case: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(r.get('case'), []).append(r)
    out: List[OpCoverage] = []
    for op in catalogue:
        q = op.get('qualname')
        runtime = op.get('jit') != 'n/a'
        cc = op_to_case.get(q)
        crows = by_case.get(cc[0], []) if cc else []
        if not cc or not crows:
            out.append(OpCoverage(
                qualname=q, runtime=runtime, has_case=bool(cc),
                coverage=UNMEASURED, ref_strength=NO_REF,
                precision='unmeasured', provisional=False))
            continue
        gpu_ref, gpu_ratio = _gpu_bar(crows, cc[1])
        out.append(OpCoverage(
            qualname=q, runtime=runtime, has_case=True,
            coverage=_coverage_status(crows),
            ref_strength=_ref_strength(crows),
            precision=_precision(crows),
            provisional=any(
                (r.get('provenance') or {}).get('coverage_mode') == 'fast'
                for r in crows),
            gpu_ref=gpu_ref, gpu_ref_ratio=gpu_ratio))
    return out


def _priority(oc: OpCoverage) -> Optional[str]:
    '''Priority tier for the under-covered list, or ``None`` if the op is fully
    covered (multiplatform + a strong on-target ref) or a host-side constructor
    (tracked separately).  A coarse, transparent heuristic -- consumer-traffic
    weighting is a future input (mandate §2.2/§4).'''
    if not oc.runtime:
        return None
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
        'gpu_ref': r.gpu_ref, 'gpu_ref_ratio': r.gpu_ref_ratio,
        'nitrix_slower_on_gpu': r.nitrix_slower_on_gpu,
    }


def _lagging(records: List[OpCoverage]) -> List[OpCoverage]:
    return sorted((r for r in records if r.nitrix_slower_on_gpu),
                  key=lambda r: r.gpu_ref_ratio or 0.0)


def _under(records: List[OpCoverage]) -> List[Tuple[OpCoverage, str]]:
    pairs = [(r, _priority(r)) for r in records if _priority(r)]
    return sorted(pairs, key=lambda rp: (
        {'high': 0, 'medium': 1}[rp[1]], rp[0].qualname))


def render_json(records: List[OpCoverage]) -> Dict[str, Any]:
    '''Machine-readable artifact for the nitrix agent (the ranked deficits).'''
    runtime = [r for r in records if r.runtime]

    def _n(pred: Any) -> int:
        return sum(1 for r in runtime if pred(r))

    lagging, under = _lagging(records), _under(records)
    return {
        'source': 'nitrix-perf-bench coverage-&-deficit report',
        'convention': 'gpu_ref_ratio = strong_ref.min / nitrix.min '
                      '(<1 => nitrix slower on the GPU)',
        'summary': {
            'runtime_ops': len(runtime),
            'measured': _n(lambda r: r.coverage != UNMEASURED),
            'multiplatform': _n(lambda r: r.coverage == MULTIPLATFORM),
            'with_strong_gpu_ref': _n(lambda r: r.ref_strength == STRONG_REF),
            'lagging_on_gpu': len(lagging),
            'constructors': sum(1 for r in records if not r.runtime),
        },
        'lagging': [_op_json(r) for r in lagging],
        'under_covered': [{**_op_json(r), 'priority': p} for r, p in under],
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


def render_markdown(records: List[OpCoverage]) -> str:
    '''Human-facing report: the two ranked lists + a coverage summary.'''
    runtime = [r for r in records if r.runtime]
    meas = sum(1 for r in runtime if r.coverage != UNMEASURED)
    multi = sum(1 for r in runtime if r.coverage == MULTIPLATFORM)
    strong = sum(1 for r in runtime if r.ref_strength == STRONG_REF)
    lagging, under = _lagging(records), _under(records)
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
        f'- **lagging on the GPU**: {len(lagging)}',
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
        '- Host-side constructors (jit `n/a`) are excluded from the runtime '
        'denominator; they have no device-time bar.',
        '',
    ]
    return '\n'.join(lines) + '\n'

