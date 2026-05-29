# -*- coding: utf-8 -*-
"""The coverage-&-deficit report (report/coverage.py; MANDATE §2.2).

Classification + the two ranked lists, on synthetic rows joined to a synthetic
catalogue -- no measurement / GPU needed.  Covers: coverage status, reference
strength, the apples-to-apples GPU bar, the lagging list (nitrix slower than
its strong on-target ref, ranked worst-first), the under-covered priorities,
the provisional (fast-run) flag, and that host-side constructors are excluded
from the runtime denominator.
"""
from nperf.report import coverage as cov

_REP = {'n': 1}
_CATALOGUE = [
    {'qualname': 'pkg.winner', 'jit': 'pass'},
    {'qualname': 'pkg.laggard', 'jit': 'pass'},
    {'qualname': 'pkg.floor_only', 'jit': 'pass'},
    {'qualname': 'pkg.cpu_only', 'jit': 'pass'},
    {'qualname': 'pkg.gpu_blocked', 'jit': 'pass'},
    {'qualname': 'pkg.unmeasured', 'jit': 'pass'},
    {'qualname': 'pkg.constructor', 'jit': 'n/a'},
]
_OP2CASE = {
    'pkg.winner': ('winner', _REP),
    'pkg.laggard': ('laggard', _REP),
    'pkg.floor_only': ('floor_only', _REP),
    'pkg.cpu_only': ('cpu_only', _REP),
    'pkg.gpu_blocked': ('blk', _REP),
    # pkg.unmeasured + pkg.constructor: no case
}


def _row(case, baseline, framework, platform, *, ratio=None, cmode='full'):
    r = {
        'case': case, 'baseline': baseline, 'framework': framework,
        'platform': platform, 'status': 'ok', 'param_point': dict(_REP),
        'metrics': {'steady_time': {'min': 1.0}},
        'provenance': {'coverage_mode': cmode},
    }
    if ratio is not None:
        r['ratio'] = {'vs': 'nitrix-jax', 'metric': 'min', 'value': ratio}
    return r


def _skip(case, baseline, framework, platform, reason):
    return {
        'case': case, 'baseline': baseline, 'framework': framework,
        'platform': platform, 'status': 'skipped', 'param_point': dict(_REP),
        'failure_detail': {'reason': reason},
        'provenance': {'coverage_mode': 'full'},
    }


def _rows():
    return [
        # winner: cupy 5x nitrix on GPU => nitrix faster
        _row('winner', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('winner', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('winner', 'numpy.x', 'numpy', 'jax-cuda12', ratio=20.0),
        _row('winner', 'cupy.x', 'cupy', 'jax-cuda12', ratio=5.0),
        # laggard: cupy 0.1x nitrix on GPU => nitrix ~10x slower
        _row('laggard', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('laggard', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('laggard', 'cupy.y', 'cupy', 'jax-cuda12', ratio=0.1),
        # floor_only: multiplatform but only a numpy CPU-floor ref
        _row('floor_only', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('floor_only', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('floor_only', 'numpy.z', 'numpy', 'jax-cuda12', ratio=30.0),
        # cpu_only: nitrix measured only on CPU
        _row('cpu_only', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('cpu_only', 'numpy.w', 'numpy', 'jax-cpu', ratio=2.0),
        # gpu_blocked: nitrix GPU skipped (backend_unavailable) but a strong
        # GPU ref (cupy) ran ok -> GPU-capable, nitrix-blocked
        _row('blk', 'nitrix-jax', 'jax', 'jax-cpu'),
        _skip('blk', 'nitrix-jax', 'jax', 'jax-cuda12', 'backend_unavailable'),
        _row('blk', 'cupy.blk', 'cupy', 'jax-cuda12'),  # ran ok (no ratio)
    ]


def _build(rows=None):
    return cov.build_coverage(rows or _rows(), _CATALOGUE, _OP2CASE)


def test_classification():
    recs = {r.qualname: r for r in _build()}
    assert recs['pkg.winner'].coverage == cov.MULTIPLATFORM
    assert recs['pkg.winner'].ref_strength == cov.STRONG_REF
    assert recs['pkg.winner'].gpu_ref == 'cupy.x'
    assert recs['pkg.winner'].nitrix_slower_on_gpu is False
    assert recs['pkg.laggard'].nitrix_slower_on_gpu is True
    assert recs['pkg.floor_only'].ref_strength == cov.FLOOR_ONLY
    assert recs['pkg.floor_only'].gpu_ref is None       # no strong GPU ref
    assert recs['pkg.cpu_only'].coverage == cov.CPU_ONLY
    assert recs['pkg.unmeasured'].coverage == cov.UNMEASURED
    assert recs['pkg.unmeasured'].has_case is False
    assert recs['pkg.constructor'].runtime is False


def test_lagging_and_summary_json():
    doc = cov.render_json(_build())
    assert [d['qualname'] for d in doc['lagging']] == ['pkg.laggard']
    s = doc['summary']
    assert s['runtime_ops'] == 6 and s['constructors'] == 1
    assert s['multiplatform'] == 3            # winner, laggard, floor_only
    assert s['with_strong_gpu_ref'] == 3      # winner, laggard, gpu_blocked
    assert s['lagging_on_gpu'] == 1
    assert s['gpu_blocked_upstream'] == 1     # pkg.gpu_blocked


def test_under_covered_priorities():
    under = {d['qualname']: d['priority']
             for d in cov.render_json(_build())['under_covered']}
    assert under['pkg.unmeasured'] == 'high'
    assert under['pkg.cpu_only'] == 'high'
    assert under['pkg.floor_only'] == 'medium'   # multiplatform, no GPU bar
    assert 'pkg.winner' not in under             # covered + winning
    assert 'pkg.laggard' not in under            # covered (it's in lagging)
    assert 'pkg.constructor' not in under        # not a runtime op


def test_provisional_from_fast_run():
    rows = _rows()
    for r in rows:
        if r['case'] == 'laggard':
            r['provenance']['coverage_mode'] = 'fast'
    recs = {r.qualname: r for r in _build(rows)}
    assert recs['pkg.laggard'].provisional is True
    assert recs['pkg.winner'].provisional is False


def test_gpu_blocked_detection():
    recs = {r.qualname: r for r in _build()}
    b = recs['pkg.gpu_blocked']
    assert b.gpu_blocked is True
    assert b.gpu_block_reason == 'backend_unavailable'  # the recorded reason
    assert b.ref_strength == cov.STRONG_REF             # cupy ran ok on GPU
    doc = cov.render_json(_build())
    assert [d['qualname'] for d in doc['gpu_blocked']] == ['pkg.gpu_blocked']
    # blocked ops are tracked separately, not in the under-covered list
    assert 'pkg.gpu_blocked' not in {
        d['qualname'] for d in doc['under_covered']}


def test_config_skip_is_not_gpu_blocked():
    # a deliberate skip (skipped_by_config) is voluntary, not an upstream
    # block -- it must NOT read as gpu_blocked even with a working GPU ref.
    rows = _rows()
    for r in rows:
        if r['case'] == 'blk' and r.get('status') == 'skipped':
            r['failure_detail']['reason'] = 'skipped_by_config'
    recs = {r.qualname: r for r in _build(rows)}
    assert recs['pkg.gpu_blocked'].gpu_blocked is False


def test_markdown_renders():
    md = cov.render_markdown(_build())
    assert 'coverage & deficit' in md
    assert 'Lagging on the deployment target' in md
    assert 'pkg.laggard' in md
    assert 'GPU blocked' in md and 'pkg.gpu_blocked' in md
