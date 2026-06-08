# -*- coding: utf-8 -*-
"""Scaling report: surface the size / scale **crossover** + HBM growth per op.

The defence against *scale-gaming* (the size-axis analogue of B18's
dispatch-branch / accuracy gaming): a perf win at a small benched size is
meaningless if a worse asymptotic or memory growth loses -- or OOMs -- before
the scale practitioners actually run.  A win at one size can be "won" by
picking a small size; the only honest answer is the **scaling curve**, plus the
stated cost *law* so the crossover is predictable, not whack-a-mole with sizes.

For each op that declares a size tier (``Case.large_param_points``) this reads
the accumulated L4 store and reports, across the dev + brain-scale sweep:

- the speed **crossover** -- the smallest total size where the best *exact*
  baseline overtakes nitrix (the ratio nitrix/baseline crosses 1.0);
- the **HBM multiplier** (nitrix peak / best-baseline peak) and a **projected
  OOM** size from the per-element memory rate vs a device budget -- nitrix vs
  the baseline (the gap is the batched-cohort headroom you actually have);
- **OOM-as-signal** -- any point where nitrix OOMed / skipped while a baseline
  ran (a first-class outcome, elevated, not a hidden row);
- the op's stated ``complexity`` law.

Reads the store only (no measurement / GPU)::

    JAX_PLATFORMS=cpu python tools/scaling_report.py
    JAX_PLATFORMS=cpu python tools/scaling_report.py --platform jax-cuda12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402

# Device HBM budget for the projected-OOM estimate (L4 = 24 GB; the headline
# device).  Overridable; only the *ratio* of nitrix-vs-baseline projections is
# device-independent, so the absolute projection is a guide, not a guarantee.
_HBM_BUDGET_MB = 24_000.0


def _prod(xs: List[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _size_elems(param: Dict[str, Any]) -> int:
    '''Total elements of a param point: prod(spatial) * batch (a single scale
    axis the curve sorts on -- batching and grid size both grow it).'''
    return _prod(param.get('shape', [])) * int(param.get('batch', 1) or 1)


def _label(param: Dict[str, Any]) -> str:
    shp = 'x'.join(str(s) for s in param.get('shape', []))
    b = param.get('batch')
    base = f'{b}*{shp}' if b else shp
    # Distinguish same-shape points by their structuring element / dtype, so a
    # box and a disk/ball at one grid size are *separate* rows -- not merged
    # (the morphology family has several SEs per shape; collapsing them hid the
    # fast box behind the slow ball).
    tags = []
    if param.get('se'):
        k = param.get('size', param.get('radius'))
        tags.append(f'{param["se"]}{k}' if k is not None else str(param['se']))
    if param.get('dtype') and param['dtype'] != 'float32':
        tags.append(str(param['dtype']))
    return base + (' ' + ','.join(tags) if tags else '')


def _steady_min(row: Dict[str, Any]) -> Optional[float]:
    m = (row.get('metrics') or {}).get('steady_time') or {}
    return m.get('min')


def _hbm(row: Dict[str, Any]) -> Optional[float]:
    m = (row.get('metrics') or {}).get('peak_hbm') or {}
    return m.get('value')


def _point_key(param: Dict[str, Any]) -> str:
    return _label(param)


def _collect(rows: List[dict], case, platform: str) -> Dict[str, Dict]:
    '''Group one case's rows (on ``platform``) by param point -> the nitrix-jax
    entry, the best *exact* baseline entry, and the raw nitrix status.'''
    approx = {a.baseline for a in case.approximate_baselines}
    pts: Dict[str, Dict] = {}
    for r in rows:
        if r.get('case') != case.name or r.get('platform') != platform:
            continue
        key = _point_key(r['param_point'])
        p = pts.setdefault(key, {
            'param': r['param_point'], 'size': _size_elems(r['param_point']),
            'nitrix': None, 'nitrix_status': None, 'baselines': [],
        })
        name = r['baseline']
        if name == 'nitrix-jax':
            p['nitrix_status'] = r.get('status')
            if r.get('status') == 'ok':
                p['nitrix'] = r
        elif name not in approx and not name.startswith('nitrix'):
            if r.get('status') == 'ok' and _steady_min(r) is not None:
                p['baselines'].append(r)
    return pts


def _best_baseline(entries: List[dict]) -> Optional[dict]:
    cand = [(e, _steady_min(e)) for e in entries if _steady_min(e) is not None]
    return min(cand, key=lambda t: t[1])[0] if cand else None


def _fmt_ms(s: Optional[float]) -> str:
    return f'{s * 1e3:.2f}ms' if s is not None else '—'


def _fmt_mb(v: Optional[float]) -> str:
    return f'{v:.1f}MB' if v is not None else '—'


def _analyse(case, pts: Dict[str, Dict]) -> Dict[str, Any]:
    '''Build the per-size table + crossover + HBM projection + OOM list.'''
    rows = []
    for p in sorted(pts.values(), key=lambda d: d['size']):
        nt = _steady_min(p['nitrix']) if p['nitrix'] else None
        base = _best_baseline(p['baselines'])
        bt = _steady_min(base) if base else None
        nh = _hbm(p['nitrix']) if p['nitrix'] else None
        bh = _hbm(base) if base else None
        rows.append({
            'label': _label(p['param']), 'size': p['size'],
            'nitrix_t': nt, 'base_t': bt,
            'base_name': base['baseline'].split('.')[-1] if base else None,
            'ratio': (nt / bt) if (nt and bt) else None,
            'nitrix_hbm': nh, 'base_hbm': bh,
            'hbm_mult': (nh / bh) if (nh and bh) else None,
            'nitrix_status': p['nitrix_status'],
        })

    # Speed: the ratio is NOT a function of total elements alone -- it depends
    # on shape/dimensionality (a 2-D EDT crosses earlier than a 3-D one of the
    # same element count, and a batch behaves like its per-image shape), so we
    # do *not* assert a single contiguous win-window or one crossover point.
    # Instead list the sizes where the baseline is ahead (the scale-gaming
    # risks), worst-first, and the win count.  HBM growth (below) *is* roughly
    # elements-linear, so its projection is the robust headline.
    ranked = [r for r in rows if r['ratio'] is not None]
    wins = [r for r in ranked if r['ratio'] <= 1.0]
    losses = sorted((r for r in ranked if r['ratio'] > 1.0),
                    key=lambda r: -r['ratio'])
    # The verdict point: the largest measured size, tie-broken to the *worst*
    # ratio (the binding scale risk when several SEs share a size).
    largest = (max(ranked, key=lambda r: (r['size'], r['ratio'] or 0))
               if ranked else None)

    # Projected OOM: the device budget over the per-element HBM rate at the
    # **heaviest measured allocation** (the binding point -- the disk/ball SE,
    # at a large size where the fixed allocator overhead is amortised; a small
    # point's rate is inflated by that fixed cost and must not drive the
    # projection).  Linear (HBM ~ O(elements)); a guide, not a guarantee.
    def _proj(kind: str) -> Optional[float]:
        sized = [r for r in rows
                 if r[kind] is not None and r[kind] > 0 and r['size'] > 0]
        if not sized:
            return None
        heavy = max(sized, key=lambda r: r[kind])
        return _HBM_BUDGET_MB / (heavy[kind] / heavy['size'])  # elements

    # OOM-as-signal: nitrix oom/skipped while a baseline ran.
    ooms = [r for r in rows
            if r['nitrix_status'] in ('oom', 'skipped') and r['base_t']]
    return {
        'rows': rows, 'wins': wins, 'losses': losses, 'largest': largest,
        'n_ranked': len(ranked),
        'proj_oom_nitrix': _proj('nitrix_hbm'),
        'proj_oom_base': _proj('base_hbm'),
        'ooms': ooms,
    }


def _render(case, platform: str, a: Dict[str, Any]) -> List[str]:
    out = [f'## {case.name}  ({case.op_qualname})  [{platform}]', '']
    if case.complexity:
        out += [f'**Cost law.** {case.complexity}', '']
    out += ['| size | nitrix | best baseline | ratio (nx/base) | nitrix HBM | '
            'base HBM | HBM x |',
            '|---|---|---|---|---|---|---|']
    for r in a['rows']:
        ratio = (f'{r["ratio"]:.2f}x' if r['ratio'] is not None
                 else (r['nitrix_status'] or '—'))
        base = f'{_fmt_ms(r["base_t"])} ({r["base_name"]})' if r['base_name'] \
            else '—'
        mult = f'{r["hbm_mult"]:.0f}x' if r['hbm_mult'] is not None else '—'
        out.append('| ' + ' | '.join([
            r['label'], _fmt_ms(r['nitrix_t']), base, ratio,
            _fmt_mb(r['nitrix_hbm']), _fmt_mb(r['base_hbm']), mult,
        ]) + ' |')
    out.append('')

    wins, losses, big, n = (a['wins'], a['losses'], a['largest'],
                            a['n_ranked'])
    if n:
        line = f'- **Speed:** nitrix wins {len(wins)}/{n} sizes'
        if losses:
            worst = ', '.join(f'`{r["label"]}` {r["ratio"]:.2f}x'
                              for r in losses[:4])
            more = '' if len(losses) <= 4 else f' (+{len(losses) - 4} more)'
            line += f'; baseline ahead at {worst}{more}'
        if big is not None:
            v = ('baseline ' + f'{big["ratio"]:.2f}x ahead'
                 if (big['ratio'] or 0) > 1.0
                 else f'nitrix {1 / big["ratio"]:.2f}x ahead')
            line += f'; at the largest `{big["label"]}`, {v}'
        out.append(line + '.')
    pn, pb = a['proj_oom_nitrix'], a['proj_oom_base']
    if pn:
        msg = (f'- **Projected OOM (≈{_HBM_BUDGET_MB / 1000:.0f}GB):** nitrix '
               f'~{pn / 1e6:.1f} Melem')
        if pb:
            msg += (f' vs best baseline ~{pb / 1e6:.0f} Melem '
                    f'(~{pb / pn:.0f}x more headroom)')
        out.append(msg + '.')
    for r in a['ooms']:
        out.append(f'- **OOM/skip-as-signal:** nitrix `{r["nitrix_status"]}` '
                   f'at `{r["label"]}` while {r["base_name"]} ran '
                   f'({_fmt_ms(r["base_t"])}).')
    out.append('')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='L4 rows: .jsonl files or store dirs (default: '
                         'store). Newest row per key is used.')
    ap.add_argument('--platform', default='jax-cuda12',
                    help='platform whose curve to analyse (default jax-cuda12 '
                         '-- the HBM/crossover story lives on the GPU)')
    ap.add_argument('--out-md', default='reports/SCALING.md')
    args = ap.parse_args()

    rows: list = []
    for f in store.expand_inputs(args.inputs):
        rows.extend(read_jsonl(f))
    rows = store.latest(rows)

    cases = [c for c in CASES.values() if c.large_param_points]
    doc = ['# Scaling / crossover report', '',
           'Scale-gaming defence: the scaling curve + the stated cost law, so '
           'a small-size win cannot hide a large-size / batched loss or OOM. '
           f'Platform: `{args.platform}`.', '']
    n_risk = 0
    for case in sorted(cases, key=lambda c: c.name):
        pts = _collect(rows, case, args.platform)
        if not pts:
            continue
        a = _analyse(case, pts)
        big = a['largest']
        pn, pb = a['proj_oom_nitrix'], a['proj_oom_base']
        # A scale risk = nitrix OOMs / loses at the largest measured size, OR
        # carries materially less memory headroom (projected OOM) than the
        # baseline (the HBM-hog risk a speed-only view misses).
        hbm_risk = bool(pn and pb and pn < 0.5 * pb)
        if a['ooms'] or (big and (big['ratio'] or 0) > 1.0) or hbm_risk:
            n_risk += 1
        doc += _render(case, args.platform, a)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text('\n'.join(doc) + '\n')
    print(f'scaling: {len(cases)} tiered op(s), {n_risk} with a scale risk '
          f'(crossover / OOM / HBM headroom) on {args.platform}. '
          f'Wrote {args.out_md}.', file=sys.stderr)


if __name__ == '__main__':
    main()
