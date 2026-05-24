# -*- coding: utf-8 -*-
"""Decision-input bundles (L5 artifact; DESIGN §1/§5).

A bundle packages the *structured inputs* a human needs to make a
"benchmark-first" call for **one op at one decision point** — the competing
baselines' ratios, each one's fidelity structure + threshold check, and the
historical trend across runs — and **emits no recommendation**.  The verdict
("JAX-default" vs "pursue Pallas") is a human-curated layer on top; until there
is enough decision history to calibrate, the suite must not let "the benchmark
said so" (a noisy crossing) drive engineering (DESIGN §1).

No new *metric* arithmetic (SCHEMA §G): the ratios and fidelity are already
computed and stored in L1; this selects the point, groups by platform, and
orders the per-run history (oldest→newest) for the trend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import store
from .core import METRICS
from .measure import CASES


def _same_point(row_pt: Dict[str, Any], rep: Dict[str, Any]) -> bool:
    return all(row_pt.get(k) == v for k, v in rep.items() if k != 'seed')


def _baseline_view(row: Dict[str, Any]) -> Dict[str, Any]:
    '''The per-baseline evidence, read straight from the stored row.'''
    m = row.get('metrics') or {}
    st = m.get('steady_time') or {}
    ratio = row.get('ratio') or {}
    fid = row.get('fidelity') or {}
    return {
        'status': row.get('status'),
        'steady': ({k: st.get(k) for k in ('min', 'median', 'p95')}
                   if st else None),
        'steady_unit': st.get('unit'),
        'ratio_vs_reference': ratio.get('value'),
        'fidelity': ({'status': fid.get('status'),
                      'rel_to_tol': fid.get('rel_to_tol')} if fid else None),
        # The fidelity gate is rel_to_tol <= 1 (== fidelity.status 'pass'); we
        # surface the already-decided result, not a re-check.
        'within_fidelity_threshold': (
            (fid.get('status') == 'pass') if fid else None),
        'peak_hbm_mb': (m.get('peak_hbm') or {}).get('value'),
        'host_rss_mb': (m.get('host_rss') or {}).get('value'),
    }


def build_bundle(
    rows: List[Dict[str, Any]], *, case: str,
    point: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    '''Assemble the decision-input bundle for ``case`` at ``point``.

    ``rows`` are L4 rows across one or more runs (e.g. the store); the current
    state per ``(platform, baseline)`` is the newest row, and the *trend* is
    each baseline's per-run history (oldest→newest) of ``min`` and ratio.'''
    c = CASES[case]
    rep = point or c.representative
    sel = [r for r in rows
           if r.get('case') == case
           and _same_point(r.get('param_point') or {}, rep)]
    current = store.latest(sel)

    ref = next((r['ratio']['vs'] for r in current if r.get('ratio')), None)

    platforms: Dict[str, Any] = {}
    for r in current:
        plat = r.get('platform')
        platforms.setdefault(plat, {'baselines': {}, 'trend': {}})
        platforms[plat]['baselines'][r.get('baseline')] = _baseline_view(r)

    # Trend: every run's OK rows, oldest→newest (run_id is timestamp-leading).
    for r in sorted(sel, key=lambda x: x.get('run_id') or ''):
        if r.get('status') != 'ok':
            continue
        plat = r.get('platform')
        st = (r.get('metrics') or {}).get('steady_time') or {}
        platforms.setdefault(plat, {'baselines': {}, 'trend': {}})
        platforms[plat]['trend'].setdefault(r.get('baseline'), []).append({
            'run_id': r.get('run_id'),
            'min': st.get('min'),
            'ratio_vs_reference': (r.get('ratio') or {}).get('value'),
        })

    return {
        'op': c.op_qualname,
        'case': case,
        'point': {k: v for k, v in rep.items() if k != 'seed'},
        'ratio_reference': ref,
        'fidelity_threshold': METRICS['fidelity'].threshold,
        'fidelity_threshold_unit': METRICS['fidelity'].unit,
        'platforms': platforms,
        # Deliberately no verdict (DESIGN §1/§5) -- present for shape, always
        # None so a downstream reader can't mistake the bundle for advice.
        'recommendation': None,
        'note': ('Decision-input bundle: structured evidence only.  The '
                 'recommendation (e.g. "JAX-default" vs "pursue Pallas") is a '
                 'human call layered on top -- the suite emits no verdict '
                 '(DESIGN §1/§5).'),
    }
