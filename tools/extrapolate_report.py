# -*- coding: utf-8 -*-
"""Extrapolation report: GUESS brain-scale from a dense small-fast-scale sweep.

The combinatorial space of modelling paths (e.g. GLMM family x structure x
method x level-count tier) is too large to probe densely at brain scale, and
the looped-CPU gold standards (R mgcv / lme4, statsmodels) cannot run there.
So we measure each path across a dense range of small, fast scales and
*extrapolate*: fit the empirical power law, check its exponent against the
declared theoretical :class:`~nperf.cases._base.CostLaw`, and project to brain
scale -- bracketing the empirical fit against a theory-anchored slope, and
when a measured ``large_param_points`` anchor exists validating the projection
against it.  The baseline's law is extrapolated too, so the brain-scale
**speedup over a competitor that is infeasible to run there** is itself a
prediction.

This complements ``tools/scaling_report.py`` (which tabulates *measured* points
+ a linear HBM projection); it is the extrapolation layer.  Op-agnostic: every
case that declares a ``cost_law`` or ``scale_paths`` is covered.  Reads the
store only (no measurement / GPU)::

    JAX_PLATFORMS=cpu python tools/extrapolate_report.py
    JAX_PLATFORMS=cpu python tools/extrapolate_report.py --platform jax-cpu

Caveats (printed in the report): a single-exponent power law assumes a clean
asymptotic regime within the swept range -- at small ``n`` fixed allocator /
dispatch overhead inflates the constant, so the empirical exponent can
undershoot theory until the range spans an order of magnitude (R^2 + theory
bracket show this).  Each path extrapolates ONE axis (others at sweep value);
the brain-scale headline that varies several axes at once is the product of the
per-axis laws, validated axis-by-axis.  Time only (HBM keeps scaling_report's
linear projection).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402

# A projection within this factor of the measured anchor (either direction) is
# `validated`; beyond it the law does not extrapolate -> `off` (a finding).
_ANCHOR_TOL = 2.0
# No anchor for a path -> project this many x past the largest swept scale (so
# every path still gets a brain-scale guess; marked `provisional`).
_TARGET_FACTOR = 100.0
# |empirical exponent - theoretical exponent| beyond this is flagged (the
# small-n regime, or a genuine surprise).
_THEORY_TOL = 0.3


def _steady_min(row: Dict[str, Any]) -> Optional[float]:
    m = (row.get('metrics') or {}).get('steady_time') or {}
    return m.get('min')


def _canon(param: Dict[str, Any]) -> str:
    return json.dumps(param, sort_keys=True)


def _fmt_t(s: Optional[float]) -> str:
    '''Seconds -> a human duration that stays readable across the ~6 orders of
    magnitude a small-to-brain projection spans.'''
    if s is None:
        return '—'
    if s < 1e-3:
        return f'{s * 1e6:.1f}us'
    if s < 1.0:
        return f'{s * 1e3:.2f}ms'
    if s < 60.0:
        return f'{s:.2f}s'
    return f'{s / 60.0:.1f}min'


def _fit_loglog(pts: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
    '''Fit ``t = a * n^alpha`` (log-log least squares) over (n, t) points.

    Duplicate ``n`` (several SEs / reps at one size) collapse to their best
    (min) steady time, matching the curve scaling_report draws.  Returns
    ``None`` if fewer than 3 distinct sizes (a power law needs a slope + a
    curvature check).  ``r2`` + ``resid_std`` (log-space) quantify the fit.'''
    agg: Dict[float, List[float]] = defaultdict(list)
    for n, t in pts:
        if n and n > 0 and t and t > 0:
            agg[float(n)].append(float(t))
    ns = np.array(sorted(agg))
    if len(ns) < 3:
        return None
    ts = np.array([min(agg[n]) for n in ns])
    lx, ly = np.log(ns), np.log(ts)
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        'a': float(math.exp(intercept)), 'alpha': float(slope), 'r2': r2,
        'resid_std': float(math.sqrt(ss_res / len(lx))), 'n_pts': int(len(ns)),
        'n_last': float(ns[-1]), 't_last': float(ts[-1]),
        'points': list(zip((float(n) for n in ns), (float(t) for t in ts))),
    }


def _paths_of(case) -> List[Tuple[Optional[str], Any, bool]]:
    '''(label, CostLaw, challenging) per modelling path.  Multi-path ->
    ``scale_paths``; single-path -> one synthetic path (``label=None`` matches
    every row) from ``cost_law``.'''
    if case.scale_paths:
        return [(p.label, p.cost, p.challenging) for p in case.scale_paths]
    if case.cost_law:
        return [(None, case.cost_law, False)]
    return []


def _collect(rows: List[dict], case, platform: str, label: Optional[str],
             axis: str, small_keys: set, large_keys: set):
    '''This path's rows -> (nitrix small, nitrix anchor, {base: small},
    {base: anchor}) as (n, t) lists keyed off the path's scale axis.'''
    approx = {a.baseline for a in case.approximate_baselines}
    nit_s: List[Tuple[float, float]] = []
    nit_a: List[Tuple[float, float]] = []
    base_s: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    base_a: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r.get('case') != case.name or r.get('platform') != platform:
            continue
        p = r['param_point']
        if label is not None and p.get('path') != label:
            continue
        if axis not in p:
            continue
        key = _canon(p)
        is_small, is_anchor = key in small_keys, key in large_keys
        if not (is_small or is_anchor):
            continue
        n, t, name = p[axis], _steady_min(r), r['baseline']
        if r.get('status') != 'ok' or not t:
            continue
        if name == 'nitrix-jax':
            (nit_a if is_anchor else nit_s).append((n, t))
        elif name not in approx and not name.startswith('nitrix'):
            (base_a if is_anchor else base_s)[name].append((n, t))
    return nit_s, nit_a, base_s, base_a


def _dominant(
        base_small: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
    '''The baseline with the most distinct sizes (the most complete curve to
    extrapolate -- usually the looped gold standard at every point).'''
    if not base_small:
        return None
    return max(base_small,
               key=lambda k: len({n for n, _ in base_small[k]}))


def _analyse_path(label, cost, challenging, nit_s, nit_a, base_s,
                  anchor_tol: float, target_factor: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'label': label, 'axis': cost.axis, 'cost': cost,
        'challenging': challenging, 'fit': None,
    }
    fit = _fit_loglog(nit_s)
    out['fit'] = fit
    if not fit:
        return out
    # brain-scale target on this axis: the measured anchor if present, else a
    # generic order-of-magnitude reach past the largest swept point.
    if nit_a:
        nstar = max(n for n, _ in nit_a)
        out['t_meas'] = min(t for n, t in nit_a if n == nstar)
    else:
        nstar = fit['n_last'] * target_factor
        out['t_meas'] = None
    out['nstar'] = nstar
    out['t_emp'] = fit['a'] * nstar ** fit['alpha']
    out['t_theory'] = fit['t_last'] * (nstar / fit['n_last']) ** cost.time_exp
    band = math.exp(fit['resid_std'])
    out['band'] = (out['t_emp'] / band, out['t_emp'] * band)
    out['theory_gap'] = abs(fit['alpha'] - cost.time_exp)
    if out['t_meas']:
        ratio = out['t_emp'] / out['t_meas']
        out['proj_ratio'] = ratio
        ok = (1 / anchor_tol) <= ratio <= anchor_tol
        out['verdict'] = 'validated' if ok else 'off'
    else:
        out['verdict'] = 'provisional'
    # baseline law -> projected brain-scale speedup (the baseline is infeasible
    # to run at nstar, so the speedup is itself an extrapolation).
    bname = _dominant(base_s)
    out['base_name'] = bname
    bfit = _fit_loglog(base_s[bname]) if bname else None
    out['base_fit'] = bfit
    if bfit:
        b_emp = bfit['a'] * nstar ** bfit['alpha']
        out['base_emp'] = b_emp
        out['speedup'] = b_emp / out['t_emp']
    return out


def _render(case, platform: str, analyses: List[Dict[str, Any]]) -> List[str]:
    out = [f'## {case.name}  ({case.op_qualname})  [{platform}]', '']
    if case.complexity:
        out += [f'**Cost law (prose).** {case.complexity}', '']
    for a in analyses:
        tag = '  `[challenging]`' if a['challenging'] else ''
        head = (f'### path: `{a["label"]}`{tag}' if a['label']
                else '### single-path')
        out += [head, '']
        c = a['cost']
        regime = f' ({c.regime})' if c.regime else ''
        out.append(f'**Theory.** axis `{c.axis}`, time ~ `{c.axis}^'
                   f'{c.time_exp:g}`{regime}.')
        fit = a['fit']
        if not fit:
            out += ['', '- _insufficient points (<3 sizes) to fit a law._', '']
            continue
        bname = a['base_name']
        out += ['', f'| {c.axis} | nitrix | {bname or "—"} |',
                '|---|---|---|']
        bpts = dict(a['base_fit']['points']) if a['base_fit'] else {}
        for n, t in fit['points']:
            out.append(f'| {n:g} | {_fmt_t(t)} | '
                       f'{_fmt_t(bpts.get(n))} |')
        gap = a['theory_gap']
        verdict = a['verdict']
        out += [
            '',
            f'- **Empirical fit:** `t ~ {fit["a"]:.2e}·{c.axis}^'
            f'{fit["alpha"]:.2f}`  (R²={fit["r2"]:.3f}, {fit["n_pts"]} pts).',
            f'- **Theory check:** empirical α={fit["alpha"]:.2f} vs theory '
            f'α={c.time_exp:g} (Δ={gap:.2f} — '
            f'{"DIVERGES" if gap > _THEORY_TOL else "agrees"}).',
        ]
        proj = (f'- **Brain-scale projection @ {c.axis}={a["nstar"]:g}:** '
                f'empirical {_fmt_t(a["t_emp"])} '
                f'[{_fmt_t(a["band"][0])}–{_fmt_t(a["band"][1])} 1σ]; '
                f'theory-anchored {_fmt_t(a["t_theory"])}.')
        if a['t_meas'] is not None:
            proj += (f'  Measured anchor {_fmt_t(a["t_meas"])}, '
                     f'proj/meas {a["proj_ratio"]:.2f}× → **{verdict}**.')
        else:
            proj += f'  **{verdict}** (no measured anchor).'
        out.append(proj)
        if a.get('speedup') is not None:
            out.append(
                f'- **Projected speedup over {bname} @ brain scale:** '
                f'~{a["speedup"]:.1f}× ({bname} is infeasible to run there; '
                f'extrapolated from its α={a["base_fit"]["alpha"]:.2f}).')
        out.append('')
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='store rows: .jsonl files or store dirs '
                         '(default: store). Newest row per key is used.')
    ap.add_argument('--platform', default='jax-cuda12',
                    help='platform whose curve to extrapolate '
                         '(default jax-cuda12).')
    ap.add_argument('--out-md', default='reports/EXTRAPOLATION.md')
    ap.add_argument('--anchor-tol', type=float, default=_ANCHOR_TOL,
                    help='projection within this factor of the measured '
                         f'anchor is `validated` (default {_ANCHOR_TOL}).')
    ap.add_argument('--target-factor', type=float, default=_TARGET_FACTOR,
                    help='no anchor: project this many x past the largest '
                         f'swept scale (default {_TARGET_FACTOR}).')
    args = ap.parse_args()

    rows: list = []
    for f in store.expand_inputs(args.inputs):
        rows.extend(read_jsonl(f))
    rows = store.latest(rows)

    cases = [c for c in CASES.values() if c.cost_law or c.scale_paths]
    doc = ['# Extrapolation report', '',
           'Brain-scale GUESS from a dense small-fast-scale sweep: per '
           'modelling path, the empirical power law + the theoretical '
           'CostLaw, projected to brain scale and (where measured) validated '
           f'against the anchor.  Platform: `{args.platform}`.  `validated` = '
           f'projection within {args.anchor_tol:g}x of the measured anchor; '
           '`off` = the law does not extrapolate (a finding); `provisional` = '
           'no anchor yet.', '']
    n_paths = n_val = n_off = 0
    for case in sorted(cases, key=lambda c: c.name):
        analyses = []
        for label, cost, challenging in _paths_of(case):
            small_keys = {_canon(p) for p in case.param_points}
            large_keys = {_canon(p) for p in case.large_param_points}
            nit_s, nit_a, base_s, _ = _collect(
                rows, case, args.platform, label, cost.axis,
                small_keys, large_keys)
            a = _analyse_path(label, cost, challenging, nit_s, nit_a,
                              base_s, args.anchor_tol, args.target_factor)
            if a['fit'] is None and not nit_s:
                continue  # no data for this path on this platform
            analyses.append(a)
            n_paths += 1
            n_val += a.get('verdict') == 'validated'
            n_off += a.get('verdict') == 'off'
        if analyses:
            doc += _render(case, args.platform, analyses)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text('\n'.join(doc) + '\n')
    print(f'extrapolation: {len(cases)} case(s) with a cost model, {n_paths} '
          f'path(s) on {args.platform} ({n_val} validated, {n_off} off-law). '
          f'Wrote {args.out_md}.', file=sys.stderr)


if __name__ == '__main__':
    main()
