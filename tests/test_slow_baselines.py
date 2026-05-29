# -*- coding: utf-8 -*-
"""The slow-baseline guard (COVERAGE_MANDATE §7).

``--skip-slow`` drops a case's declared slow baselines for fast dev cycles.
The load-bearing property: a *fast* run never masquerades as authoritative
coverage.  Two protections are tested:

- the gate does not false-fail when a current row is a deliberately-skipped
  baseline (omission != slowdown), but a genuine ``ok -> oom`` still regresses;
- the op_matrix feed ignores rows from a fast run (only full runs bless it).
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

from nperf import gate
from nperf.core import Status
from nperf.measure import CASES
from nperf.run import run_case_inprocess

_FEED = Path(__file__).resolve().parents[1] / 'tools' / 'op_matrix_feed.py'
_spec = importlib.util.spec_from_file_location('op_matrix_feed', _FEED)
feed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feed)


def test_semiring_matmul_declares_naive_dense_slow():
    names = {s.baseline for s in CASES['semiring_matmul'].slow_baselines}
    assert 'naive-dense' in names


def test_skip_slow_labels_slow_skipped():
    # ``slow`` drops + labels 'slow_skipped', distinct from an explicit skip's
    # 'skipped_by_config', so the coverage layer can tell them apart.
    recs = run_case_inprocess(
        replace(CASES['cov'], param_points=[{'c': 8, 'n_obs': 16}]),
        platform='jax-cpu', warmup=1, repeats=2, prov={}, run_id='t',
        slow=frozenset({'numpy.cov'}),
    )
    by = {r.baseline: r for r in recs}
    assert by['nitrix-jax'].status == Status.OK
    assert by['numpy.cov'].status == Status.SKIPPED
    assert by['numpy.cov'].failure_detail == {'reason': 'slow_skipped'}


def _gate_row(baseline, mn, *, status='ok', run_id='r'):
    r = {'case': 'semiring_matmul', 'platform': 'jax-cuda12',
         'baseline': baseline, 'status': status, 'run_id': run_id,
         'param_point': {'m': 512, 'algebra': 'log'}}
    if mn is not None:
        r['metrics'] = {'steady_time': {'min': mn, 'p95': mn}}
    return r


def test_gate_skipped_current_is_not_a_regression():
    base = [_gate_row('naive-dense', 1.0, run_id='r1')]
    curr = [_gate_row('naive-dense', None, status='skipped', run_id='r2')]
    art = gate.compare(base, curr)
    e = art['comparisons'][0]
    assert e['kind'] == 'skipped_current' and e['regressed'] is False
    assert art['summary']['verdict'] == 'pass'


def test_gate_genuine_status_regression_still_fails():
    base = [_gate_row('naive-dense', 1.0, run_id='r1')]
    curr = [_gate_row('naive-dense', None, status='oom', run_id='r2')]
    art = gate.compare(base, curr)
    e = art['comparisons'][0]
    assert e['kind'] == 'status_regression' and e['regressed'] is True
    assert art['summary']['verdict'] == 'fail'


def _feed_row(coverage_mode, mn):
    return {'case': 'semiring_matmul', 'platform': 'jax-cuda12',
            'baseline': 'nitrix-jax', 'status': 'ok',
            'param_point': {'m': 512, 'algebra': 'log'},
            'metrics': {'steady_time': {'min': mn}},
            'provenance': {'coverage_mode': coverage_mode}}


def test_feed_ignores_fast_rows():
    full = _feed_row('full', 1.0)
    fast = _feed_row('fast', 2.0)
    legacy = _feed_row('full', 3.0)
    del legacy['provenance']  # pre-guard run -> treated as full
    kept = feed.authoritative_rows([full, fast, legacy])
    assert full in kept and legacy in kept
    assert fast not in kept
