# -*- coding: utf-8 -*-
"""Decision-input bundles (bundle.py + report/bundle.py; DESIGN §1/§5).

The bundle packages evidence (ratios, fidelity, trend) for one op at one point
and emits NO recommendation.  Current state = newest row per (platform,
baseline); trend = per-run history, oldest->newest.  No new arithmetic -- every
value is read from the rows.
"""
from nperf import bundle
from nperf.core import METRICS
from nperf.measure import CASES
from nperf.report import render_bundle

_REP = CASES['semiring_matmul'].representative  # 512^3 log


def _row(baseline, mn, ratio, *, run_id, status='ok', platform='jax-cpu',
         fidelity_status='pass', rel_to_tol=0.1):
    r = {
        'case': 'semiring_matmul', 'platform': platform, 'baseline': baseline,
        'status': status, 'run_id': run_id,
        'param_point': dict(_REP),
    }
    if status == 'ok':
        r['metrics'] = {
            'steady_time': {'min': mn, 'median': mn * 1.1, 'p95': mn * 1.2,
                            'unit': 's'},
            'peak_hbm': {'value': 5.0, 'unit': 'MB'},
        }
        if baseline != 'nitrix-jax':
            r['ratio'] = {'vs': 'nitrix-jax', 'metric': 'min', 'value': ratio}
        r['fidelity'] = {'status': fidelity_status, 'rel_to_tol': rel_to_tol}
    return r


def test_bundle_groups_current_state_and_reference():
    rows = [
        _row('nitrix-jax', 1.0, None, run_id='20260101T000000__a'),
        _row('nitrix-pallas', 0.2, 0.2, run_id='20260101T000000__a'),
    ]
    b = bundle.build_bundle(rows, case='semiring_matmul')
    assert b['op'] == 'nitrix.semiring.semiring_matmul'
    assert b['ratio_reference'] == 'nitrix-jax'
    assert b['fidelity_threshold'] == METRICS['fidelity'].threshold
    bl = b['platforms']['jax-cpu']['baselines']
    assert bl['nitrix-pallas']['ratio_vs_reference'] == 0.2
    assert bl['nitrix-pallas']['within_fidelity_threshold'] is True


def test_bundle_emits_no_recommendation():
    rows = [_row('nitrix-jax', 1.0, None, run_id='20260101T000000__a')]
    b = bundle.build_bundle(rows, case='semiring_matmul')
    assert b['recommendation'] is None
    assert 'no verdict' in b['note']


def test_trend_is_per_run_oldest_first_and_current_is_newest():
    rows = [
        _row('nitrix-pallas', 0.30, 0.30, run_id='20260101T000000__a'),
        _row('nitrix-pallas', 0.18, 0.18, run_id='20260202T000000__b'),
    ]
    b = bundle.build_bundle(rows, case='semiring_matmul')
    trend = b['platforms']['jax-cpu']['trend']['nitrix-pallas']
    assert [t['ratio_vs_reference'] for t in trend] == [0.30, 0.18]
    # current state is the newest run.
    cur = b['platforms']['jax-cpu']['baselines']['nitrix-pallas']
    assert cur['ratio_vs_reference'] == 0.18


def test_failed_baseline_surfaced_without_steady():
    rows = [
        _row('nitrix-jax', 1.0, None, run_id='r'),
        _row('nitrix-pallas', None, None, run_id='r', status='oom'),
    ]
    b = bundle.build_bundle(rows, case='semiring_matmul')
    bp = b['platforms']['jax-cpu']['baselines']['nitrix-pallas']
    assert bp['status'] == 'oom' and bp['steady'] is None


def test_render_contains_evidence_and_no_verdict_note():
    rows = [
        _row('nitrix-jax', 1.0, None, run_id='r'),
        _row('nitrix-pallas', 0.2, 0.2, run_id='r'),
    ]
    md = render_bundle(bundle.build_bundle(rows, case='semiring_matmul'))
    assert 'Decision-input bundle' in md
    assert 'nitrix-pallas' in md and '0.20×' in md
    assert 'no verdict' in md
