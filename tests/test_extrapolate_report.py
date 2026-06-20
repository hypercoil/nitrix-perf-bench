# -*- coding: utf-8 -*-
"""Extrapolation fitter + projection math (``tools/extrapolate_report.py``).

The methodology stands or falls on the fit: feed synthetic rows with a PLANTED
power law and assert (a) the recovered exponent + constant match the plant, (b)
the brain-scale projection equals the closed form, (c) the anchor verdict flips
validated/off/provisional as the measured anchor agrees or not, and (d) the
projected speedup over a steeper baseline grows out to brain scale.  No store,
no GPU -- pure arithmetic, so the math is pinned independently of any run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import extrapolate_report as er  # noqa: E402

from nperf.cases._base import (  # noqa: E402
    Case,
    CostLaw,
    ScalePath,
    scaling_sweep,
)


def _row(baseline, param, t, case='synth', platform='jax-cpu'):
    return {'case': case, 'baseline': baseline, 'platform': platform,
            'param_point': param, 'status': 'ok',
            'metrics': {'steady_time': {'min': t}}}


def test_fit_recovers_planted_exponent():
    """log-log fit recovers a planted ``t = a·n^alpha`` and its projection
    equals the closed form."""
    a0, alpha0 = 1e-6, 1.5
    pts = [(n, a0 * n ** alpha0) for n in (10, 32, 100, 316, 1000)]
    fit = er._fit_loglog(pts)
    assert abs(fit['alpha'] - alpha0) < 1e-9
    assert fit['r2'] > 0.9999
    assert abs(fit['a'] - a0) / a0 < 1e-6
    nstar = 1e5
    closed = a0 * nstar ** alpha0
    proj = fit['a'] * nstar ** fit['alpha']
    assert abs(proj - closed) / closed < 1e-6


def test_fit_needs_three_distinct_sizes():
    """A power law needs >=3 distinct sizes (slope + a curvature check);
    duplicate sizes collapse and do not count."""
    assert er._fit_loglog([(10, 1.0), (100, 2.0)]) is None
    assert er._fit_loglog([(10, 1.0), (10, 1.1), (100, 2.0)]) is None
    assert er._fit_loglog([(10, 1.0), (100, 2.0), (1000, 3.0)]) is not None


def test_analyse_validates_projects_and_brackets_speedup():
    """``_analyse_path``: planted nitrix (α=1) + a steeper baseline (α=2);
    anchor agreement -> validated, projected speedup grows to brain scale."""
    cl = CostLaw('n', 1.0, regime='linear batch')
    nit_s = [(n, 1e-6 * n) for n in (10, 100, 1000)]            # α 1, a 1e-6
    base_s = {'r.looped': [(n, 1e-7 * n ** 2) for n in (10, 100, 1000)]}  # α 2
    nstar = 1e5

    # measured anchor == closed-form projection -> validated
    a = er._analyse_path('p', cl, False, nit_s, [(nstar, 1e-6 * nstar)],
                         base_s, 2.0, 100.0)
    assert a['verdict'] == 'validated'
    assert abs(a['nstar'] - nstar) < 1
    assert abs(a['t_emp'] - 0.1) / 0.1 < 1e-3        # 1e-6 * 1e5
    assert a['theory_gap'] < 1e-6                     # empirical α == theory α
    # baseline α=2 -> brain-scale speedup ~ (1e-7*1e10) / 0.1 = 1e4
    assert a['base_name'] == 'r.looped'
    assert abs(a['speedup'] - 1e4) / 1e4 < 1e-2

    # anchor far from projection -> off (a finding: the law mis-extrapolates)
    off = er._analyse_path('p', cl, False, nit_s, [(nstar, 10.0)],
                           base_s, 2.0, 100.0)
    assert off['verdict'] == 'off'

    # no anchor -> provisional, target = target_factor x largest swept size
    prov = er._analyse_path('p', cl, False, nit_s, [], base_s, 2.0, 100.0)
    assert prov['verdict'] == 'provisional'
    assert abs(prov['nstar'] - 100.0 * 1000) < 1


def test_theory_gap_flags_exponent_divergence():
    """When the empirical exponent disagrees with the declared CostLaw, the gap
    exceeds the flag threshold (the small-n / surprise signal)."""
    cl = CostLaw('q', 3.0)                            # theory cubic ...
    nit_s = [(n, 1e-6 * n) for n in (10, 100, 1000)]  # ... but measured linear
    a = er._analyse_path('p', cl, False, nit_s, [], {}, 2.0, 100.0)
    assert a['theory_gap'] > er._THEORY_TOL


def test_collect_partitions_small_vs_anchor_and_filters_path():
    """``_collect`` groups one path's rows by axis value, splits small-sweep vs
    anchor (by param membership), excludes other paths, and keeps each exact
    baseline -- so a multi-axis case stays cleanly separated."""
    p1 = ScalePath('p1', {'fam': 'a'}, CostLaw('n', 1.0), (10, 100, 1000))
    p2 = ScalePath('p2', {'fam': 'b'}, CostLaw('q', 1.0), (8, 16))
    pp = scaling_sweep([p1, p2])
    large = [{'fam': 'a', 'n': 100000, 'path': 'p1'}]
    case = Case(name='synth', output_independent=True, metrics=['steady_time'],
                param_points=pp, representative=pp[0], build=lambda p: None,
                op_qualname='x.synth', scale_paths=(p1, p2),
                large_param_points=tuple(large))
    small_keys = {er._canon(p) for p in case.param_points}
    large_keys = {er._canon(p) for p in case.large_param_points}

    rows = []
    for p in pp:
        if p['path'] == 'p1':
            rows.append(_row('nitrix-jax', p, 1e-6 * p['n']))
            rows.append(_row('r.looped', p, 1e-7 * p['n'] ** 2))
    rows.append(_row('nitrix-jax', large[0], 1e-6 * 1e5))

    nit_s, nit_a, base_s, _ = er._collect(
        rows, case, 'jax-cpu', 'p1', 'n', small_keys, large_keys)
    assert sorted(n for n, _ in nit_s) == [10, 100, 1000]
    assert len(nit_a) == 1 and nit_a[0][0] == 100000
    assert len(base_s['r.looped']) == 3

    # p2 has no rows -> empty (axes don't bleed across paths)
    nit_s2, _, _, _ = er._collect(
        rows, case, 'jax-cpu', 'p2', 'q', small_keys, large_keys)
    assert nit_s2 == []
