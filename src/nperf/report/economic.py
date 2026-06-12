# -*- coding: utf-8 -*-
"""Economic verdict core (shared L5).

The cross-platform join behind ``tools/economic_report.py`` AND the coverage
report's *economic* axis: is the nitrix-GPU win **multiplicative** over the CPU
gold standard -- enough to clear the GPU:CPU cost-multiple bar?  A pure store
read (no measurement, no metric arithmetic -- every figure is a stored L1 row).

Factored out of the tool so the report (ECONOMIC.md) and the coverage matrix
share ONE verdict function.  The tool keeps its renderer; this module owns the
join (``analyse``), the label (``verdict``), and the per-op reduction
(``op_verdict``) the coverage axis needs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .sizing import label, size_elems

# GPU:CPU cost multiple -- an L4 GPU-hour costs ~4x an equivalent CPU-hour on
# the major clouds (an L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs
# a comparable general-purpose vCPU instance ~$0.18/hr, 2026). A nitrix-GPU
# result is only "economically favorable" if it beats the CPU gold standard by
# MORE than this -- an incremental GPU win is not a win once the user pays the
# GPU premium. Tunable via --cost-multiple.
COST_MULTIPLE = 4.0
_GPU = 'jax-cuda12'
_CPU = 'jax-cpu'


def _steady_min(row: Dict[str, Any]) -> Optional[float]:
    return ((row.get('metrics') or {}).get('steady_time') or {}).get('min')


def _compile(row: Dict[str, Any]) -> Optional[float]:
    return ((row.get('metrics') or {}).get('compile_time') or {}).get('value')


def _pkey(param: Dict[str, Any]) -> str:
    return json.dumps(param, sort_keys=True)


# Input-realism ladder (low->high): synthetic data+truth; real DATA with a
# synthetic/known ground truth (a planted warp on a real image -- recoverable
# truth); and the full-realism ACTUAL problem (real data, no planted truth).
REALISM_RUNGS = ('synthetic', 'real_planted', 'real_full')


def realism_rung(param: Dict[str, Any]) -> str:
    '''The realism rung of a param point (convention).  An explicit
    ``regime``/``realism`` tag wins; else a ``'data'`` marker (the registration
    real-anatomy convention -- real image + a *planted* warp) reads as
    ``real_planted``; else ``synthetic``.'''
    tag = param.get('regime') or param.get('realism')
    if tag in ('real_full', 'real_problem'):
        return 'real_full'
    if tag in ('real_planted', 'real') or 'data' in param:
        return 'real_planted'
    return 'synthetic'


def rung_index(rung: str) -> int:
    return REALISM_RUNGS.index(rung) if rung in REALISM_RUNGS else 0


def verdict(amortized: Optional[float], single: Optional[float],
            bar: float) -> str:
    '''The economic label for one point at the cost-multiple ``bar``.'''
    if amortized is None:
        return 'n/a'
    if amortized < bar:
        # a real GPU win can still be NOT a win once the hardware premium is
        # paid -- the user's central point.
        return 'not multiplicative enough'
    if single is not None and single >= bar:
        return 'favorable'
    return 'favorable (amortized only)'  # the compile is the gate


def analyse(case: Any, rows: List[dict], bar: float,
            points: Optional[List[Dict[str, Any]]] = None
            ) -> List[Dict[str, Any]]:
    '''Join GPU-nitrix x CPU-gold per param point for one case.

    ``points`` defaults to the case **size tier** (``large_param_points``):
    that is where the scale-relevant comparison lives, and -- crucially -- the
    CPU domain tools (ANTs/dipy) run a FIXED schedule that ignores the dev
    ``(levels, iters)`` configs, so a verdict on those would be meaningless.
    The coverage axis passes explicit ``points`` (the representative) for the
    non-authoritative fallback.'''
    pts = points if points is not None else list(case.large_param_points)
    keys = {_pkey(p) for p in pts}
    if not keys:
        return []
    by_key: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get('case') != case.name:
            continue
        k = _pkey(r['param_point'])
        if k in keys:
            by_key.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for grp in by_key.values():
        gpu = [r for r in grp if r['baseline'] == 'nitrix-jax'
               and r['platform'] == _GPU and r['status'] == 'ok']
        if not gpu or _steady_min(gpu[0]) is None:
            continue
        gs, gc = _steady_min(gpu[0]), _compile(gpu[0])
        cpu_ok = [r for r in grp if r['platform'] == _CPU
                  and r['status'] == 'ok' and _steady_min(r) is not None]
        # I/O floors by provider namespace (the name prefix before the dot):
        # a CLI tool's wall-clock includes a NIfTI write+subprocess+read the
        # in-memory nitrix op never pays -- subtract the same-namespace no-op
        # (afni.iofloor / fsl.iofloor) to isolate the registration COMPUTE.
        floors = {r['baseline'].split('.')[0]: _steady_min(r)
                  for r in cpu_ok if r['baseline'].endswith('.iofloor')}

        def _compute(r: dict) -> float:
            f = floors.get(r['baseline'].split('.')[0], 0.0)
            return max(_steady_min(r) - f, 1e-6)  # I/O-subtracted

        domain = [r for r in cpu_ok if r['baseline'] != 'nitrix-jax'
                  and not r['baseline'].endswith('.iofloor')]
        if domain:  # strongest competitor = fastest tool AFTER I/O subtraction
            best = min(domain, key=_compute)
            tool, fallback = best['baseline'], False
            cpu_raw = _steady_min(best)
            iofloor = floors.get(tool.split('.')[0])
            cpu = _compute(best)
        else:  # no domain tool (e.g. BBR) -> GPU-vs-own-CPU (no I/O artifact)
            nc = [r for r in cpu_ok if r['baseline'] == 'nitrix-jax']
            if not nc:
                continue
            tool, fallback = 'nitrix-CPU', True
            cpu_raw, iofloor = _steady_min(nc[0]), None
            cpu = cpu_raw
        amort = cpu / gs if gs and gs > 0 else None
        single = (cpu / (gs + gc)) if (gc is not None and (gs + gc) > 0
                                       ) else None
        param = gpu[0]['param_point']
        out.append({
            'label': label(param), 'size': size_elems(param), 'param': param,
            'realism': realism_rung(param),
            'gpu_steady': gs, 'gpu_compile': gc, 'cpu': cpu, 'tool': tool,
            'cpu_raw': cpu_raw, 'iofloor': iofloor,
            'fallback': fallback, 'amort': amort, 'single': single,
            'verdict': verdict(amort, single, bar),
        })
    out.sort(key=lambda d: d['size'])
    return out


@dataclass(frozen=True)
class OpVerdict:
    '''The single economic verdict for an op (the coverage-matrix cell).'''
    verdict: str                 # favorable | favorable (amortized only) |
    #                              not multiplicative enough | n/a | unmeasured
    amortized: Optional[float]
    single: Optional[float]
    authoritative: bool          # measured at a real/large point (vs the
    #                              representative fallback)
    point: Optional[str]         # the label of the point the verdict is from


def op_verdict(case: Any, rows: List[dict], bar: float = COST_MULTIPLE
               ) -> OpVerdict:
    '''Reduce a case to ONE economic verdict for the coverage matrix
    (decision 4): the **largest real point** if measured, else the **largest
    large point**, else the **representative** point flagged
    ``authoritative=False``; ``unmeasured`` if no GPU-vs-CPU join exists.

    "Is nitrix even on GPU here?" is decided by the caller (a GPU-blocked op is
    ``n/a``, not ``unmeasured``); this only reduces an existing join.'''
    large = analyse(case, rows, bar)
    if large:
        # the most-real measured point, then largest within that rung
        r = max(large, key=lambda d: (rung_index(d['realism']), d['size']))
        return OpVerdict(r['verdict'], r['amort'], r['single'],
                         authoritative=True, point=r['label'])
    rep = analyse(case, rows, bar, points=[case.representative])
    if rep:
        r = rep[0]
        return OpVerdict(r['verdict'], r['amort'], r['single'],
                         authoritative=False, point=r['label'])
    return OpVerdict('unmeasured', None, None, authoritative=False, point=None)
