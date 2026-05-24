# -*- coding: utf-8 -*-
"""The regression gate (gate.py + report/gate.py; SCHEMA §F).

Dual signal on steady_time: min (tight) and p95 (loose); a key trips if either
ratio exceeds its threshold.  Status transitions are first-class (an ok->fail
is a regression with no ratio; the reverse is not).  Keys present on one side
are reported but do not fail the gate.
"""
from nperf import gate
from nperf.report import render_gate


def _row(baseline, mn, p95, *, status='ok', platform='jax-cpu',
         case='c', algebra='real'):
    r = {
        'case': case, 'platform': platform, 'baseline': baseline,
        'status': status, 'run_id': '20260101T000000__x',
        'param_point': {'m': 8, 'k': 8, 'n': 8, 'algebra': algebra},
    }
    if mn is not None:
        r['metrics'] = {'steady_time': {'min': mn, 'p95': p95, 'unit': 's'}}
    return r


def test_clean_pass():
    base = [_row('nitrix-jax', 1.0, 1.2)]
    curr = [_row('nitrix-jax', 1.02, 1.22)]  # within both thresholds
    art = gate.compare(base, curr)
    assert art['summary']['verdict'] == 'pass'
    assert art['comparisons'][0]['regressed'] is False


def test_min_trip_fails():
    base = [_row('nitrix-jax', 1.0, 1.2)]
    curr = [_row('nitrix-jax', 1.20, 1.25)]  # min 1.2x > 1.10
    art = gate.compare(base, curr)
    assert art['summary']['verdict'] == 'fail'
    e = art['comparisons'][0]
    assert e['trips']['min'] and e['ratios']['min'] == 1.2


def test_p95_trip_only_fails():
    # min (1.05x < 1.10) but p95 balloons (1.5x > 1.25): a shape regression.
    base = [_row('nitrix-jax', 1.0, 1.0)]
    curr = [_row('nitrix-jax', 1.05, 1.5)]
    art = gate.compare(base, curr)
    e = art['comparisons'][0]
    assert e['regressed'] and e['trips']['p95'] and not e['trips']['min']


def test_custom_thresholds():
    base = [_row('nitrix-jax', 1.0, 1.0)]
    curr = [_row('nitrix-jax', 1.20, 1.2)]
    # A generous min threshold (1.5) lets the 1.2x min pass.
    art = gate.compare(base, curr, min_threshold=1.5, p95_threshold=1.5)
    assert art['summary']['verdict'] == 'pass'
    assert art['thresholds'] == {'min': 1.5, 'p95': 1.5}


def test_status_regression_fails_without_ratio():
    base = [_row('nitrix-jax', 1.0, 1.2)]
    curr = [_row('nitrix-jax', None, None, status='oom')]
    art = gate.compare(base, curr)
    e = art['comparisons'][0]
    assert e['kind'] == 'status_regression' and e['regressed']
    assert art['summary']['verdict'] == 'fail'


def test_recovery_is_not_a_regression():
    base = [_row('nitrix-jax', None, None, status='compile_error')]
    curr = [_row('nitrix-jax', 1.0, 1.2)]
    art = gate.compare(base, curr)
    e = art['comparisons'][0]
    assert e['kind'] == 'recovered' and not e['regressed']
    assert art['summary']['verdict'] == 'pass'


def test_new_and_dropped_keys_reported_not_failed():
    base = [_row('nitrix-jax', 1.0, 1.2)]
    curr = [_row('nitrix-jax', 1.0, 1.2), _row('pyg', 0.5, 0.6)]  # pyg is new
    art = gate.compare(base, curr)
    kinds = {(e['baseline']): e['kind'] for e in art['comparisons']}
    assert kinds['pyg'] == 'new'
    assert art['summary']['n_new'] == 1
    assert art['summary']['verdict'] == 'pass'

    # And the reverse: a key only in baseline is 'dropped'.
    art2 = gate.compare(base + [_row('gone', 1.0, 1.2)], curr)
    dropped = [e for e in art2['comparisons'] if e['kind'] == 'dropped']
    assert len(dropped) == 1 and not dropped[0]['regressed']


def test_regressions_helper_and_render():
    base = [_row('nitrix-jax', 1.0, 1.2), _row('pyg', 1.0, 1.0)]
    curr = [_row('nitrix-jax', 1.5, 1.6), _row('pyg', 1.0, 1.0)]
    art = gate.compare(base, curr)
    regs = gate.regressions(art)
    assert len(regs) == 1 and regs[0]['baseline'] == 'nitrix-jax'
    md = render_gate(art)
    assert 'FAIL' in md and 'nitrix-jax' in md and '## Regressions' in md
