# -*- coding: utf-8 -*-
"""Regression-gate renderer (L5).

Presents the gate artifact (`nperf.gate.compare`) as markdown.  Does **no**
arithmetic — the min/p95 ratios and the verdict are already computed and stored
in the artifact (SCHEMA §G); this only selects, orders, and formats.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .markdown import _fmt_param

# Order rows so the gate's drivers are at the top, then the rest.
_KIND_ORDER = {
    'status_regression': 0, 'compared': 1, 'still_failing': 2,
    'recovered': 3, 'dropped': 4, 'new': 5,
}


def _ratio_cell(e: Dict[str, Any], stat: str) -> str:
    ratios = e.get('ratios') or {}
    r = ratios.get(stat)
    if r is None:
        return '—'
    flag = ' ⚠️' if (e.get('trips') or {}).get(stat) else ''
    return f'{r:.2f}×{flag}'


def _row_label(e: Dict[str, Any]) -> str:
    return (f"{e['case']} | {e['platform']} | {_fmt_param(e['param_point'])} "
            f"| `{e['baseline']}`")


def _kind_note(e: Dict[str, Any]) -> str:
    kind = e.get('kind')
    if kind == 'status_regression':
        return f"**status {e['baseline_status']} → {e['current_status']}**"
    if kind == 'recovered':
        return f"recovered ({e['baseline_status']} → {e['current_status']})"
    if kind == 'still_failing':
        return f"still {e['current_status']}"
    if kind == 'new':
        return 'new (no baseline)'
    if kind == 'dropped':
        return 'dropped (no current)'
    return ''


def render_gate(artifact: Dict[str, Any]) -> str:
    th = artifact['thresholds']
    s = artifact['summary']
    verdict = s['verdict']
    badge = '🟢 PASS' if verdict == 'pass' else '🔴 FAIL'
    lines: List[str] = [
        '# Regression gate',
        '',
        f'**{badge}** — {s["n_regressed"]} regressed of {s["n_compared"]} '
        f'compared ({s["n_keys"]} keys; {s["n_new"]} new, '
        f'{s["n_dropped"]} dropped).',
        '',
        f'Dual signal on `steady_time` (SCHEMA §F): a key trips if '
        f'**min** ratio > {th["min"]}× (tight, best-case slowdown) **or** '
        f'**p95** ratio > {th["p95"]}× (loose, distribution-shape). Ratio = '
        f'current / baseline; ⚠️ marks the tripping statistic.',
        '',
        '| case | platform | param | baseline | min | p95 | note |',
        '|---|---|---|---|---|---|---|',
    ]
    ordered = sorted(
        artifact['comparisons'],
        key=lambda e: (not e.get('regressed'),
                       _KIND_ORDER.get(e.get('kind'), 9),
                       e['case'], e['platform'], e['baseline']),
    )
    for e in ordered:
        lines.append(
            f"| {e['case']} | {e['platform']} | "
            f"{_fmt_param(e['param_point'])} | `{e['baseline']}` | "
            f"{_ratio_cell(e, 'min')} | {_ratio_cell(e, 'p95')} | "
            f"{_kind_note(e)} |"
        )
    if verdict == 'fail':
        drivers = [_row_label(e) for e in ordered if e.get('regressed')]
        lines += ['', '## Regressions', '']
        lines += [f'- {d}' for d in drivers]
    lines.append('')
    return '\n'.join(lines)
