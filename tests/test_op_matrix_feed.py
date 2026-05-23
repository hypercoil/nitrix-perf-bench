# -*- coding: utf-8 -*-
"""The op_matrix perf feed (tools/op_matrix_feed.py).

Checks the ratio convention (primary.min / reference.min; <1 = nitrix faster),
the platform->device mapping, and graceful handling when the reference is
missing -- on synthetic rows, so no measurement / GPU is needed.
"""
import importlib.util
from pathlib import Path

_FEED = Path(__file__).resolve().parents[1] / 'tools' / 'op_matrix_feed.py'
_spec = importlib.util.spec_from_file_location('op_matrix_feed', _FEED)
feed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feed)


def _row(platform, baseline, mn, *, status='ok', vs='nitrix-jax'):
    r = {
        'case': 'semiring_matmul', 'platform': platform, 'baseline': baseline,
        'status': status,
        'param_point': {'m': 512, 'k': 512, 'n': 512, 'algebra': 'log',
                        'seed': 0},
    }
    if status == 'ok':
        r['metrics'] = {'steady_time': {'min': mn, 'unit': 'ms'}}
        if baseline != vs:
            r['ratio'] = {'vs': vs, 'metric': 'min', 'value': mn}  # unused
    return r


def test_ratio_is_primary_over_reference_per_device():
    rows = [
        _row('jax-cpu', 'nitrix-jax', 1.0),
        _row('jax-cpu', 'naive-dense', 2.0),     # nitrix 2x faster on cpu
        _row('jax-cuda12', 'nitrix-jax', 8.0),
        _row('jax-cuda12', 'naive-dense', 1.0),  # nitrix 8x slower on gpu
    ]
    frag = feed.build_fragment(rows, 'semiring_matmul', 'naive-dense')
    e = frag['nitrix.semiring.semiring_matmul']
    assert e['perf_cpu_baseline'] == 'naive-dense'
    assert e['perf_cpu_ratio'] == 0.5    # 1.0 / 2.0
    assert e['perf_gpu_ratio'] == 8.0    # 8.0 / 1.0
    assert e['_meta']['point']['algebra'] == 'log'


def test_missing_reference_records_note_not_crash():
    rows = [_row('jax-cpu', 'nitrix-jax', 1.0)]  # no naive-dense row
    frag = feed.build_fragment(rows, 'semiring_matmul', 'naive-dense')
    e = frag['nitrix.semiring.semiring_matmul']
    assert e['perf_cpu_baseline'] is None and e['perf_cpu_ratio'] is None
    assert 'cpu' in e['_meta']['notes']


def test_non_ok_primary_yields_no_ratio():
    rows = [
        _row('jax-cpu', 'nitrix-jax', 0.0, status='oom'),
        _row('jax-cpu', 'naive-dense', 2.0),
    ]
    frag = feed.build_fragment(rows, 'semiring_matmul', 'naive-dense')
    e = frag['nitrix.semiring.semiring_matmul']
    assert e['perf_cpu_ratio'] is None


def test_alternate_reference_torch():
    rows = [
        _row('jax-cpu', 'nitrix-jax', 1.0),
        _row('jax-cpu', 'torch-dense', 4.0),
    ]
    frag = feed.build_fragment(rows, 'semiring_matmul', 'torch-dense')
    e = frag['nitrix.semiring.semiring_matmul']
    assert e['perf_cpu_baseline'] == 'torch-dense'
    assert e['perf_cpu_ratio'] == 0.25
