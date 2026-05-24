# -*- coding: utf-8 -*-
"""HTML ``/site`` renderer (L5).

A **single, self-contained** HTML page (no CDN, no build step, works offline):
per-case sortable / filterable tables plus inline-SVG plots (time-vs-size
log-log, and per-baseline history across runs).  The plots' axis scaling is a
*pure presentation transform* (log mapping to pixels) — allowed for a renderer;
every underlying number (ratios, errors) is already computed and stored in L1
(SCHEMA §G).  The rendered ``/site`` is git-ignored; this code is the artifact.

``render_site(rows)`` takes L4 rows across any runs/platforms/cases: tables
show the *current* state (newest row per key); the history plot uses every run.
"""
from __future__ import annotations

import html
import math
from typing import Any, Dict, List, Optional, Tuple

from ..store import latest
from .markdown import _fmt_param, _fmt_time

# A small categorical palette (colour-blind-ish), cycled per series.
_PALETTE = ['#4477aa', '#ee6677', '#228833', '#ccbb44', '#66ccee',
            '#aa3377', '#bbbbbb', '#000000']


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def _size(param: Dict[str, Any]) -> int:
    '''A monotonic scale scalar for the x-axis: product of the integer params
    (excluding ``seed``).  Presentation-only ordering, not a reported value.'''
    p = 1
    for k, v in param.items():
        if k == 'seed':
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            p *= v
        elif isinstance(v, (list, tuple)):  # e.g. shape=[64, 64, 64]
            for e in v:
                if isinstance(e, int) and not isinstance(e, bool):
                    p *= e
    return p


def _series_key(row: Dict[str, Any]) -> str:
    return f"{row.get('platform')} · {row.get('baseline')}"


def _steady_min(row: Dict[str, Any]) -> Optional[float]:
    st = (row.get('metrics') or {}).get('steady_time') or {}
    v = st.get('min')
    return float(v) if v is not None else None


# --------------------------------------------------------------------------- #
# Inline SVG (log-log scatter+line)                                           #
# --------------------------------------------------------------------------- #
def _svg_loglog(
    series: Dict[str, List[Tuple[float, float]]], *,
    xlabel: str, ylabel: str, width: int = 640, height: int = 360,
) -> str:
    '''Self-contained log-log SVG for ``{label: [(x, y), ...]}`` (x, y > 0).'''
    pts = [(x, y) for s in series.values() for (x, y) in s if x > 0 and y > 0]
    if not pts:
        return '<p class="muted">no plottable data</p>'
    pad_l, pad_b, pad_t, pad_r = 60, 44, 16, 150
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t
    lxs = [math.log10(x) for x, _ in pts]
    lys = [math.log10(y) for _, y in pts]
    lxmin, lxmax = min(lxs), max(lxs)
    lymin, lymax = min(lys), max(lys)
    # Guard degenerate (single distinct value) ranges.
    if lxmax - lxmin < 1e-9:
        lxmin, lxmax = lxmin - 0.5, lxmax + 0.5
    if lymax - lymin < 1e-9:
        lymin, lymax = lymin - 0.5, lymax + 0.5

    def px(x: float) -> float:
        return x0 + (math.log10(x) - lxmin) / (lxmax - lxmin) * (x1 - x0)

    def py(y: float) -> float:
        return y0 - (math.log10(y) - lymin) / (lymax - lymin) * (y0 - y1)

    out: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="plot" '
        f'role="img" aria-label="{_esc(ylabel)} vs {_esc(xlabel)}">'
    ]
    # Decade gridlines + tick labels.
    for lx in range(math.floor(lxmin), math.ceil(lxmax) + 1):
        gx = px(10 ** lx)
        if x0 <= gx <= x1:
            out.append(f'<line x1="{gx:.1f}" y1="{y1}" x2="{gx:.1f}" '
                       f'y2="{y0}" class="grid"/>')
            out.append(f'<text x="{gx:.1f}" y="{y0 + 16}" '
                       f'class="tick" text-anchor="middle">1e{lx}</text>')
    for ly in range(math.floor(lymin), math.ceil(lymax) + 1):
        gy = py(10 ** ly)
        if y1 <= gy <= y0:
            out.append(f'<line x1="{x0}" y1="{gy:.1f}" x2="{x1}" '
                       f'y2="{gy:.1f}" class="grid"/>')
            out.append(f'<text x="{x0 - 8}" y="{gy + 4:.1f}" '
                       f'class="tick" text-anchor="end">1e{ly}</text>')
    # Axes.
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" class="axis"/>')
    out.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 6}" '
               f'class="axlabel" text-anchor="middle">{_esc(xlabel)}</text>')
    out.append(f'<text x="14" y="{(y0 + y1) / 2:.0f}" class="axlabel" '
               f'text-anchor="middle" transform="rotate(-90 14 '
               f'{(y0 + y1) / 2:.0f})">{_esc(ylabel)}</text>')
    # Series: sorted points -> polyline + markers; legend at right.
    for i, (label, raw) in enumerate(sorted(series.items())):
        pts_s = sorted((x, y) for (x, y) in raw if x > 0 and y > 0)
        if not pts_s:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        poly = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in pts_s)
        out.append(f'<polyline points="{poly}" fill="none" '
                   f'stroke="{colour}" stroke-width="2"/>')
        for x, y in pts_s:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" '
                       f'fill="{colour}"><title>{_esc(label)}: '
                       f'({x:g}, {y:g})</title></circle>')
        ly = y1 + 18 * i + 6
        out.append(f'<rect x="{x1 + 12}" y="{ly - 9:.0f}" width="11" '
                   f'height="11" fill="{colour}"/>')
        out.append(f'<text x="{x1 + 28}" y="{ly:.0f}" class="legend">'
                   f'{_esc(label)}</text>')
    out.append('</svg>')
    return ''.join(out)


# --------------------------------------------------------------------------- #
# Tables + page assembly                                                      #
# --------------------------------------------------------------------------- #
_COLUMNS = ['platform', 'param', 'baseline', 'status', 'steady_min',
            'compile', 'mem', 'fidelity', 'ratio']


def _fmt_compile(row: Dict[str, Any]) -> str:
    c = (row.get('metrics') or {}).get('compile_time') or {}
    return _fmt_time(c.get('value')) if c.get('value') is not None else '—'


def _fmt_mem(row: Dict[str, Any]) -> str:
    m = row.get('metrics') or {}
    hbm = (m.get('peak_hbm') or {}).get('value')
    if hbm is not None:
        return f'{hbm:.1f} MB (hbm)'
    rss = (m.get('host_rss') or {}).get('value')
    return f'{rss:.0f} MB (rss)' if rss is not None else '—'


def _fmt_fid(row: Dict[str, Any]) -> str:
    fid = row.get('fidelity') or {}
    if not fid:
        return '—'
    mark = '✓' if fid.get('status') == 'pass' else '✗'
    r = fid.get('rel_to_tol')
    if r is None:
        return f"{mark} {fid.get('status')}"
    return f'{mark} {r:.3g}'


def _fmt_ratio(row: Dict[str, Any]) -> str:
    r = row.get('ratio') or {}
    return f"{r['value']:.2f}×" if r.get('value') is not None else '—'


def _cell(row: Dict[str, Any], col: str) -> Tuple[str, str]:
    '''(display, sort-key) for a column; numeric sort keys where it helps.'''
    if col == 'param':
        return _esc(_fmt_param(row.get('param_point') or {})), \
            str(_size(row.get('param_point') or {})).zfill(20)
    if col == 'steady_min':
        v = _steady_min(row)
        return (_fmt_time(v) if v is not None else '—',
                f'{v:.12e}' if v is not None else '')
    if col == 'compile':
        return _fmt_compile(row), ''
    if col == 'mem':
        return _fmt_mem(row), ''
    if col == 'fidelity':
        return _fmt_fid(row), ''
    if col == 'ratio':
        r = (row.get('ratio') or {}).get('value')
        return _fmt_ratio(row), f'{r:.6f}' if r is not None else ''
    return _esc(row.get(col, '')), _esc(row.get(col, ''))


def _table(rows: List[Dict[str, Any]]) -> str:
    head = ''.join(f'<th onclick="sortTable(this)">{_esc(c)}</th>'
                   for c in _COLUMNS)
    body: List[str] = []
    for row in sorted(rows, key=lambda r: (r.get('platform') or '',
                                           _size(r.get('param_point') or {}),
                                           r.get('baseline') or '')):
        tds = []
        for c in _COLUMNS:
            disp, sort = _cell(row, c)
            attr = f' data-sort="{sort}"' if sort else ''
            tds.append(f'<td{attr}>{disp}</td>')
        cls = '' if row.get('status') == 'ok' else ' class="notok"'
        body.append(f'<tr{cls}>{"".join(tds)}</tr>')
    return (f'<table class="rows"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _time_vs_size(rows: List[Dict[str, Any]]) -> str:
    series: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        if row.get('status') != 'ok':
            continue
        v = _steady_min(row)
        if v is None:
            continue
        series.setdefault(_series_key(row), []).append(
            (float(_size(row.get('param_point') or {})), v))
    return _svg_loglog(series, xlabel='size (∏ params)',
                       ylabel='steady min (s)')


def _history(all_rows: List[Dict[str, Any]]) -> str:
    '''steady-min over runs per series (only if some series has >1 run).'''
    by_series: Dict[str, Dict[str, float]] = {}
    for row in all_rows:
        if row.get('status') != 'ok':
            continue
        v = _steady_min(row)
        if v is None:
            continue
        rid = row.get('run_id') or ''
        by_series.setdefault(_series_key(row), {})[rid] = v
    if not any(len(d) > 1 for d in by_series.values()):
        return ''
    run_ids = sorted({rid for d in by_series.values() for rid in d})
    idx = {rid: i + 1 for i, rid in enumerate(run_ids)}
    series = {k: [(float(idx[rid]), v) for rid, v in sorted(d.items())]
              for k, d in by_series.items()}
    return ('<h3>History over runs</h3>'
            + _svg_loglog(series, xlabel='run (chronological)',
                          ylabel='steady min (s)'))


_STYLE = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;color:#222;background:#fafafa}
header{background:#222;color:#eee;padding:14px 20px}
header h1{margin:0;font-size:18px}
header .prov{color:#aaa;font-size:12px;margin-top:4px}
main{padding:20px;max-width:1100px;margin:0 auto}
section{background:#fff;border:1px solid #e4e4e4;border-radius:6px;
  padding:16px;margin-bottom:22px}
h2{margin:0 0 10px;font-size:16px}
.muted{color:#999}
input.filter{margin:8px 0;padding:6px 8px;width:280px;border:1px solid #ccc;
  border-radius:4px}
table.rows{border-collapse:collapse;width:100%;font-size:13px}
table.rows th,table.rows td{border-bottom:1px solid #eee;padding:5px 8px;
  text-align:left}
table.rows th{cursor:pointer;background:#f4f4f4;user-select:none;
  position:sticky;top:0}
table.rows tr.notok td{color:#999;font-style:italic}
table.rows .ok{color:#228833;font-weight:600}
table.rows .bad{color:#cc3311;font-weight:600;cursor:help}
footer{padding:8px 4px 40px;color:#444}
footer h2{font-size:15px}
footer ul{padding-left:18px}
footer li{margin:4px 0}
a{color:#3366bb}
svg.plot{max-width:100%;height:auto;background:#fff;margin:8px 0}
svg.plot .grid{stroke:#eee;stroke-width:1}
svg.plot .axis{stroke:#666;stroke-width:1}
svg.plot .tick{fill:#888;font-size:10px}
svg.plot .axlabel{fill:#444;font-size:11px}
svg.plot .legend{fill:#333;font-size:11px}
"""

_SCRIPT = """
function sortTable(th){
  var t=th.closest('table'),tb=t.tBodies[0],
      i=[].indexOf.call(th.parentNode.children,th),
      asc=!(th.dataset.asc==='1');
  th.dataset.asc=asc?'1':'0';
  var rows=[].slice.call(tb.rows);
  rows.sort(function(a,b){
    var x=(a.cells[i].dataset.sort||a.cells[i].textContent),
        y=(b.cells[i].dataset.sort||b.cells[i].textContent),
        nx=parseFloat(x),ny=parseFloat(y),
        c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:(x<y?-1:x>y?1:0);
    return asc?c:-c;});
  rows.forEach(function(r){tb.appendChild(r);});
}
function filterRows(inp){
  var q=inp.value.toLowerCase(),
      tb=inp.closest('section').querySelector('tbody');
  [].forEach.call(tb.rows,function(r){
    r.style.display=r.textContent.toLowerCase().indexOf(q)>=0?'':'none';});
}
"""


def _prov_line(prov: Dict[str, Any]) -> str:
    nitrix = (prov.get('nitrix') or {}).get('sha') or '?'
    bench = (prov.get('bench') or {}).get('sha') or '?'
    return (f"nitrix {nitrix[:10]} · bench {bench[:10]} · "
            f"{prov.get('os', '?')} · jax {prov.get('jax_version', '?')}")


def _cap_glyph(status: Any) -> str:
    '''nitrix capability-probe status -> glyph (error text kept as a title).'''
    s = str(status)
    if s == 'pass':
        return '<span class="ok">✓</span>'
    if s in ('n/a', 'not-run', 'None', ''):
        return '<span class="muted">—</span>'
    return f'<span class="bad" title="{_esc(s)}">✗</span>'


def _qual_to_case() -> Dict[str, str]:
    '''Map each benchmarked op qualname -> its case name (lazy: pulls jax).'''
    from ..measure import CASES
    return {c.op_qualname: c.name for c in CASES.values() if c.op_qualname}


def _capability_section(
    capability: Dict[str, Any], cases_present: List[Optional[str]]
) -> str:
    '''The nitrix capability matrix (jit/grad/vmap probes) as the overview,
    with a link from each op perf-bench also benchmarks to its section.'''
    ops = capability.get('ops') or []
    if not ops:
        return ''
    q2c = _qual_to_case()
    present = set(cases_present)
    host = capability.get('host') or {}
    rows_html: List[str] = []
    for op in sorted(ops, key=lambda o: o.get('qualname', '')):
        qual = op.get('qualname', '')
        case = q2c.get(qual)
        bench = (f'<a href="#{_esc(case)}">⚡ benchmarked</a>'
                 if case in present else '<span class="muted">—</span>')
        inv = '; '.join(op.get('invariants') or [])
        rows_html.append(
            f'<tr><td><code>{_esc(qual)}</code></td>'
            f'<td>{_cap_glyph(op.get("jit"))}</td>'
            f'<td>{_cap_glyph(op.get("grad"))}</td>'
            f'<td>{_cap_glyph(op.get("vmap"))}</td>'
            f'<td>{_cap_glyph(op.get("jit_of_grad"))}</td>'
            f'<td>{_esc(inv)}</td><td>{bench}</td></tr>'
        )
    cap_host = _esc(host.get('device') or host.get('platform') or '')
    return (
        '<section id="capability"><h2>Capability matrix '
        '<span class="muted">(from nitrix)</span></h2>'
        '<p class="muted">jit / grad / vmap / jit(grad) probes + invariants, '
        'sourced from nitrix\'s own op_matrix (capability lives with nitrix; '
        f'perf lives here). Probed on {cap_host}. '
        '⚡ links to this suite\'s benchmark for that op.</p>'
        '<input class="filter" placeholder="filter ops…" '
        'oninput="filterRows(this)">'
        '<table class="rows"><thead><tr>'
        '<th onclick="sortTable(this)">op</th>'
        '<th onclick="sortTable(this)">jit</th>'
        '<th onclick="sortTable(this)">grad</th>'
        '<th onclick="sortTable(this)">vmap</th>'
        '<th onclick="sortTable(this)">jit(grad)</th>'
        '<th onclick="sortTable(this)">invariants</th>'
        '<th onclick="sortTable(this)">perf</th>'
        f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table></section>'
    )


_CAVEATS = [
    'Numbers are <b>device- and env-specific</b>: ratios are computed '
    '<b>within a platform</b> (never across), on <code>min</code>, and stored '
    'in L1 — this page does no metric arithmetic.',
    '<code>steady</code> is the post-warm-up min/median; <code>compile</code> '
    'is the <b>cold</b> first-call cost (cleared cache per attempt), not a '
    'steady-state number.',
    '<code>fidelity</code> is <code>rel_to_tol</code> (tolerance-relative, '
    '✓ ⟺ ≤ 1×tol) vs an fp64 oracle — a bare relative error is meaningless '
    'for zero-centred outputs.',
    'Memory is <b>per-attempt</b> (each attempt ran in its own process): jax '
    'baselines report jax HBM, torch baselines report torch\'s allocator, CPU '
    'rows report host RSS.',
    'fp32 matmul precision is forced to <code>highest</code> (true fp32, no '
    'TF32 downgrade); the fp64 oracle runs under x64 — both recorded in '
    'provenance.',
]


def _caveats_footer() -> str:
    items = ''.join(f'<li>{c}</li>' for c in _CAVEATS)
    return (f'<footer><h2>How to read these numbers</h2><ul>{items}</ul>'
            '<p class="muted">Generated from committed L4 result rows; the '
            'page is reproducible from that data and is never hand-edited.'
            '</p></footer>')


def render_site(
    rows: List[Dict[str, Any]], *,
    capability: Optional[Dict[str, Any]] = None,
) -> str:
    '''Render the whole self-contained ``/site`` page from L4 rows.

    ``capability`` (nitrix's parsed ``op_matrix.json``) is overlaid as the
    overview when given — capability stays nitrix's, perf stays ours.'''
    current = latest(rows)
    cases = sorted({r.get('case') for r in current})
    prov = rows[0].get('provenance', {}) if rows else {}
    parts: List[str] = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>nitrix-perf-bench</title>',
        f'<style>{_STYLE}</style></head><body>',
        '<header><h1>nitrix-perf-bench</h1>',
        f'<div class="prov">{_esc(_prov_line(prov))} · '
        f'{len(current)} current rows across {len(cases)} case(s) · '
        'generated from L4 rows (no hand-edited values)</div></header><main>',
    ]
    if capability:
        parts.append(_capability_section(capability, cases))
    for case in cases:
        cur_c = [r for r in current if r.get('case') == case]
        all_c = [r for r in rows if r.get('case') == case]
        parts.append(
            f'<section id="{_esc(case)}"><h2>{_esc(case)}</h2>'
            '<input class="filter" placeholder="filter rows…" '
            'oninput="filterRows(this)">'
            f'{_table(cur_c)}'
            '<h3>steady time vs size</h3>'
            f'{_time_vs_size(cur_c)}'
            f'{_history(all_c)}'
            '</section>'
        )
    parts.append(_caveats_footer())
    parts.append(f'<script>{_SCRIPT}</script></main></body></html>')
    return ''.join(parts)
