# -*- coding: utf-8 -*-
"""Regression gate (L7 orchestration computing L1 deltas; SCHEMA §F).

Diffs a *current* run against a stored *baseline* run on **two** statistics of
``steady_time`` and fires if **either** trips for any key:

- **min** (tight threshold) — noise-robust detection of best-case slowdowns;
- **p95** (loose threshold) — **distribution-shape** regressions (a slow-path
  XLA fusion firing part of the time, or bimodality) that leave ``min``
  untouched.

The ratios *are* metric arithmetic, so they are computed **here** (L1) and
stored in the artifact; the renderer only presents them (SCHEMA §G).  Keys are
matched across runs by ``(case, platform, param_point, baseline)`` — the same
identity the store uses — after collapsing each side to its newest row per key.

Status transitions are first-class, not just numeric deltas: an ``ok`` baseline
that now fails (``status_regression``) trips the gate even though no ratio
exists; the reverse (``recovered``) does not.  Keys present on only one side
are reported (``new`` / ``dropped``) but do **not** fail the gate by default —
dropped coverage is a policy call surfaced for a human, not an automatic red.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .store import _key, latest

# Defaults: min tight (best-case must not slip >10%), p95 loose (shape may
# wobble up to 25% before it counts).  Both are stamped into every artifact.
DEFAULT_MIN_THRESHOLD = 1.10
DEFAULT_P95_THRESHOLD = 1.25


def _steady(row: Dict[str, Any], stat: str) -> Optional[float]:
    '''The ``steady_time`` ``stat`` (min / p95) of a row, or None if absent.'''
    st = (row.get('metrics') or {}).get('steady_time') or {}
    v = st.get(stat)
    return float(v) if v is not None else None


def _index(rows: List[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    '''Newest row per cross-run key (a multi-run input collapses cleanly).'''
    return {_key(r): r for r in latest(rows)}


def _compare_ok_pair(
    b: Dict[str, Any], c: Dict[str, Any],
    min_threshold: float, p95_threshold: float,
) -> Dict[str, Any]:
    '''The dual-signal numeric comparison for two ``ok`` rows.'''
    ratios: Dict[str, Optional[float]] = {}
    trips: Dict[str, bool] = {}
    for stat, thr in (('min', min_threshold), ('p95', p95_threshold)):
        bv, cv = _steady(b, stat), _steady(c, stat)
        if bv is not None and cv is not None and bv > 0:
            r = cv / bv
            ratios[stat] = round(r, 4)
            trips[stat] = r > thr
        else:
            ratios[stat] = None
            trips[stat] = False
    return {
        'kind': 'compared', 'status': 'ok',
        'ratios': ratios, 'trips': trips,
        'regressed': any(trips.values()),
    }


def compare(
    baseline_rows: List[Dict[str, Any]],
    current_rows: List[Dict[str, Any]],
    *,
    min_threshold: float = DEFAULT_MIN_THRESHOLD,
    p95_threshold: float = DEFAULT_P95_THRESHOLD,
) -> Dict[str, Any]:
    '''Build the machine-readable gate artifact (verdict + per-key deltas).'''
    base = _index(baseline_rows)
    curr = _index(current_rows)
    comparisons: List[Dict[str, Any]] = []
    for k in sorted(set(base) | set(curr)):
        case, platform, param_json, baseline = k
        b, c = base.get(k), curr.get(k)
        entry: Dict[str, Any] = {
            'case': case, 'platform': platform,
            'param_point': json.loads(param_json), 'baseline': baseline,
        }
        if b is None:
            entry.update(kind='new', regressed=False)
        elif c is None:
            entry.update(kind='dropped', regressed=False)
        else:
            bs, cs = b.get('status'), c.get('status')
            if bs == 'ok' and cs == 'ok':
                entry.update(
                    _compare_ok_pair(b, c, min_threshold, p95_threshold)
                )
            elif bs == 'ok' and cs != 'ok':
                # A deliberately-skipped current row (--skip-baselines /
                # --skip-slow) is an *omission*, not a slowdown: it must not
                # trip the gate, else a fast dev run false-fails against a full
                # baseline.  A genuine ok->{oom, compile_error, …} still
                # regresses.
                if cs == 'skipped':
                    entry.update(kind='skipped_current', baseline_status=bs,
                                 current_status=cs, regressed=False)
                else:
                    entry.update(kind='status_regression', baseline_status=bs,
                                 current_status=cs, regressed=True)
            elif bs != 'ok' and cs == 'ok':
                entry.update(kind='recovered', baseline_status=bs,
                             current_status=cs, regressed=False)
            else:
                entry.update(kind='still_failing', baseline_status=bs,
                             current_status=cs, regressed=False)
        comparisons.append(entry)

    n_regressed = sum(e['regressed'] for e in comparisons)

    def _n(kind: str) -> int:
        return sum(e.get('kind') == kind for e in comparisons)

    return {
        'thresholds': {'min': min_threshold, 'p95': p95_threshold},
        'summary': {
            'n_keys': len(comparisons),
            'n_compared': _n('compared'),
            'n_regressed': n_regressed,
            'n_new': _n('new'),
            'n_dropped': _n('dropped'),
            'verdict': 'fail' if n_regressed else 'pass',
        },
        'comparisons': comparisons,
    }


def regressions(artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
    '''The comparisons that tripped the gate (verdict drivers).'''
    return [e for e in artifact['comparisons'] if e.get('regressed')]
