# -*- coding: utf-8 -*-
"""Decision-input bundle renderer (L5).

Presents the bundle (`nperf.bundle.build_bundle`) as markdown: the competing
baselines per platform with their ratios, fidelity, and memory, plus the
per-run trend — and the standing reminder that **the suite emits no verdict**.
No arithmetic (SCHEMA §G); every number is read from the bundle.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .markdown import _fmt_param, _fmt_time


def _fmt_ratio(v: Any) -> str:
    return '—' if v is None else f'{v:.2f}×'


def _fidelity_cell(bv: Dict[str, Any]) -> str:
    fid = bv.get('fidelity')
    if not fid:
        return '—'
    status = fid.get('status')
    if status == 'inconclusive':  # no cross-impl oracle
        return 'n/a (no oracle)'
    mark = '✓' if status == 'pass' else '✗'
    r = fid.get('rel_to_tol')
    if r is None:
        return f'{mark} {status}'
    return f'{mark} {r:.3g}×tol'


def _mem_cell(bv: Dict[str, Any]) -> str:
    hbm = bv.get('peak_hbm_mb')
    if hbm is not None:
        return f'{hbm:.1f} MB (hbm)'
    rss = bv.get('host_rss_mb')
    return f'{rss:.0f} MB (rss)' if rss is not None else '—'


def _steady_cell(bv: Dict[str, Any]) -> str:
    st = bv.get('steady')
    if bv.get('status') != 'ok' or not st:
        return f"_{bv.get('status')}_"
    return (f"{_fmt_time(st.get('min'))} / {_fmt_time(st.get('median'))} "
            f"/ {_fmt_time(st.get('p95'))}")


def render_bundle(bundle: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"# Decision-input bundle — `{bundle.get('op') or bundle['case']}`",
        '',
        f"**Point:** {_fmt_param(bundle['point'])}  ·  **ratios vs** "
        f"`{bundle.get('ratio_reference')}`  ·  **fidelity gate** "
        f"rel_to_tol ≤ {bundle['fidelity_threshold']}"
        f"{bundle.get('fidelity_threshold_unit', '')}",
        '',
    ]
    for plat, pdata in sorted(bundle['platforms'].items()):
        lines += [
            f'## {plat}', '',
            '| baseline | steady (min/med/p95) | ratio | fidelity | mem |',
            '|---|---|---|---|---|',
        ]
        for b, bv in sorted(pdata['baselines'].items()):
            lines.append(
                f"| `{b}` | {_steady_cell(bv)} | "
                f"{_fmt_ratio(bv.get('ratio_vs_reference'))} | "
                f"{_fidelity_cell(bv)} | {_mem_cell(bv)} |"
            )
        lines.append('')
        multi = {b: t for b, t in pdata['trend'].items() if len(t) > 1}
        if multi:
            lines.append('**Trend** (ratio vs reference, oldest→newest):')
            for b, t in sorted(multi.items()):
                seq = ' → '.join(
                    _fmt_ratio(x.get('ratio_vs_reference')) for x in t)
                lines.append(f'- `{b}`: {seq}')
        else:
            lines.append('_Trend: single run on record — no history yet._')
        lines.append('')
    lines += ['---', '', f"> {bundle['note']}", '']
    return '\n'.join(lines)
