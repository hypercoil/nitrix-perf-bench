# -*- coding: utf-8 -*-
"""The coverage-&-deficit report (report/coverage.py; MANDATE §2.2).

Classification + the two ranked lists, on synthetic rows joined to a synthetic
catalogue -- no measurement / GPU needed.  Covers: coverage status, reference
strength, the apples-to-apples GPU bar, the lagging list (nitrix slower than
its strong on-target ref, ranked worst-first), the under-covered priorities,
the provisional (fast-run) flag, and that host-side constructors are excluded
from the runtime denominator.
"""
from types import SimpleNamespace

from nperf.report import coverage as cov
from nperf.report import economic as econ

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


def _c(name, large=(), tier='standard'):
    '''A minimal Case stand-in (build_coverage reads .name / .representative /
    .large_param_points / .tier).'''
    return SimpleNamespace(name=name, representative=_REP,
                           large_param_points=tuple(large), tier=tier)


_OP2CASE = {
    'pkg.winner': _c('winner'),
    'pkg.laggard': _c('laggard'),
    'pkg.floor_only': _c('floor_only'),
    'pkg.cpu_only': _c('cpu_only'),
    'pkg.gpu_blocked': _c('blk'),
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


def test_is_community_classification():
    # named community libraries count; our own numpy reimpl-oracles + iofloor
    # no-ops do not.
    assert cov._is_community('scipy.signal.sosfiltfilt')
    assert cov._is_community('sklearn.metrics.pairwise')
    assert cov._is_community('monai.RandGaussianNoise')
    assert cov._is_community('simpleitk.Median')      # a domain CLI
    assert not cov._is_community('numpy.intensity')    # our reimpl-oracle
    assert not cov._is_community('scipy.signal.iofloor')


_CPU_CAT = [
    {'qualname': 'pkg.cpu_lag', 'jit': 'pass'},     # scipy 4x faster on CPU
    {'qualname': 'pkg.cpu_numpy', 'jit': 'pass'},   # only a numpy reimpl ref
    {'qualname': 'pkg.cpu_win', 'jit': 'pass'},     # nitrix faster than scipy
]
_CPU_O2C = {q['qualname']: _c(q['qualname'].split('.')[-1]) for q in _CPU_CAT}


def _cpu_rows():
    return [
        # cpu_lag: a community (scipy) CPU ref 4x faster than nitrix on CPU; a
        # cupy GPU ref also runs (the lens fires independent of the GPU story).
        _row('cpu_lag', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('cpu_lag', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('cpu_lag', 'scipy.foo', 'numpy', 'jax-cpu', ratio=0.25),
        _row('cpu_lag', 'cupy.foo', 'cupy', 'jax-cuda12', ratio=5.0),
        # cpu_numpy: the only CPU ref is our numpy reimpl-oracle -> excluded,
        # so the lens abstains (cpu_ref None) even though it is 4x faster.
        _row('cpu_numpy', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('cpu_numpy', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('cpu_numpy', 'numpy.bar', 'numpy', 'jax-cpu', ratio=0.25),
        # cpu_win: scipy CPU ref is 2x slower -> nitrix ahead on CPU, no flag.
        _row('cpu_win', 'nitrix-jax', 'jax', 'jax-cpu'),
        _row('cpu_win', 'nitrix-jax', 'jax', 'jax-cuda12'),
        _row('cpu_win', 'scipy.baz', 'numpy', 'jax-cpu', ratio=2.0),
    ]


def test_cpu_community_lens():
    recs = {r.qualname: r for r in
            cov.build_coverage(_cpu_rows(), _CPU_CAT, _CPU_O2C)}
    lag = recs['pkg.cpu_lag']
    assert lag.cpu_ref == 'scipy.foo' and lag.cpu_ref_ratio == 0.25
    assert lag.cpu_gap == 4.0 and cov._lags_on_cpu(lag)
    # a numpy reimpl-oracle is not a community competitor -> abstain.
    assert recs['pkg.cpu_numpy'].cpu_ref is None
    assert recs['pkg.cpu_numpy'].cpu_gap is None
    # nitrix ahead on CPU: ref recorded, but not a lag.
    win = recs['pkg.cpu_win']
    assert win.cpu_ref == 'scipy.baz' and not cov._lags_on_cpu(win)


def test_cpu_lens_json_summary_and_threshold():
    recs = cov.build_coverage(_cpu_rows(), _CPU_CAT, _CPU_O2C)
    doc = cov.render_json(recs)
    assert doc['summary']['lagging_on_cpu_vs_community'] == 1
    assert [d['qualname'] for d in doc['cpu_lagging_vs_community']] == [
        'pkg.cpu_lag']
    # threshold is tunable: at >=5x the 4x gap no longer qualifies.
    assert cov._cpu_lagging(recs, gap=5.0) == []
    assert len(cov._cpu_lagging(recs, gap=1.5)) == 1


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


# ====================== COVERAGE v2 axes ===================================
def _vrow(baseline, platform, status, param, *, framework='jax',
          steady=None, compile=None, case='op'):
    metrics = {}
    if steady is not None:
        metrics['steady_time'] = {'min': steady}
    if compile is not None:
        metrics['compile_time'] = {'value': compile}
    return {'case': case, 'baseline': baseline, 'platform': platform,
            'status': status, 'param_point': param, 'framework': framework,
            'metrics': metrics}


# -- the ref-class fix: community gold standards are 'domain', not floor -----
def test_ref_class_domain_vs_floor_vs_strong():
    assert cov._ref_class('fsl.flameo', 'numpy') == 'domain'
    assert cov._ref_class('ants.registration', 'ants') == 'domain'
    assert cov._ref_class('statsmodels.MixedLM', 'statsmodels') == 'domain'
    # an I/O no-op in a domain namespace is a floor, NOT a reference
    assert cov._ref_class('fsl.iofloor', 'numpy') == 'floor'
    assert cov._ref_class('scipy', 'numpy') == 'floor'
    assert cov._ref_class('cupy.eigh', 'cupy') == 'strong'
    assert cov._ref_class('nitrix-jax', 'jax') == 'nitrix'


def test_realism_rungs():
    assert econ.realism_rung({'V': 100}) == 'synthetic'
    assert econ.realism_rung({'data': 'mni152'}) == 'real_planted'
    assert econ.realism_rung({'regime': 'real_full'}) == 'real_full'
    assert (econ.rung_index('real_full') > econ.rung_index('real_planted')
            > econ.rung_index('synthetic'))


def test_scale_no_tier_and_scaled():
    assert cov._scale_status(_c('op'), []) == (cov.NO_TIER, None, None)
    case = _c('op', large=[{'V': 1000}, {'V': 2000}])
    rows = [_vrow('nitrix-jax', 'jax-cpu', 'ok', {'V': 1000}),
            _vrow('nitrix-jax', 'jax-cpu', 'ok', {'V': 2000})]
    assert cov._scale_status(case, rows) == (cov.SCALED, 2000, None)


def test_scale_capped_like_flame():
    case = _c('op', large=[{'V': 1000}, {'V': 2000}])
    rows = [_vrow('nitrix-jax', 'jax-cpu', 'ok', {'V': 1000}),
            _vrow('nitrix-jax', 'jax-cpu', 'timeout', {'V': 2000})]
    assert cov._scale_status(case, rows) == (cov.SCALE_CAPPED, 1000, 'timeout')


def test_scale_declared_unmeasured():
    assert cov._scale_status(_c('op', large=[{'V': 1}]), [])[0] \
        == cov.SCALE_DECLARED


def test_input_realism_and_domain_ref():
    assert cov._input_realism(
        [_vrow('nitrix-jax', 'jax-cpu', 'ok', {'data': 'mni152'})]) \
        == 'real_planted'
    assert cov._input_realism(
        [_vrow('nitrix-jax', 'jax-cpu', 'ok', {'V': 1})]) == 'synthetic'
    rows = [_vrow('ants.registration', 'jax-cpu', 'ok', {'V': 1},
                  framework='ants'),
            _vrow('ants.registration', 'jax-cpu', 'ok', {'data': 'mni152'},
                  framework='ants')]
    assert cov._domain_ref(rows) == ('ants.registration', 'real_planted')
    assert cov._domain_ref(
        [_vrow('scipy', 'jax-cpu', 'ok', {'V': 1}, framework='numpy')]) \
        == (None, 'synthetic')


def test_economic_na_and_fallback_and_authoritative():
    # not on GPU -> n/a
    assert cov._economic(_c('op'), [], cov.CPU_ONLY, 4.0) \
        == ('n/a', None, True)
    # no large tier -> representative fallback (non-authoritative)
    case = _c('op', large=())
    rows = [_vrow('nitrix-jax', 'jax-cuda12', 'ok', {'n': 1},
                  steady=0.001, compile=0.5),
            _vrow('nitrix-jax', 'jax-cpu', 'ok', {'n': 1}, steady=0.010)]
    ov = econ.op_verdict(case, rows, bar=4.0)
    assert ov.verdict == 'favorable (amortized only)'
    assert ov.authoritative is False
    # measured at a large point vs a domain tool -> authoritative favorable
    case2 = _c('op', large=[{'V': 1000}])
    rows2 = [_vrow('nitrix-jax', 'jax-cuda12', 'ok', {'V': 1000},
                   steady=0.001, compile=0.01),
             _vrow('ants.registration', 'jax-cpu', 'ok', {'V': 1000},
                   framework='ants', steady=1.0)]
    ov2 = econ.op_verdict(case2, rows2, bar=4.0)
    assert ov2.verdict == 'favorable' and ov2.authoritative is True


# -- tier + score + marquee (Phase 2) ---------------------------------------
def _oc(**kw):
    base = dict(qualname='q', runtime=True, has_case=True,
                coverage=cov.MULTIPLATFORM, ref_strength=cov.NO_REF,
                precision='f32_only', provisional=False)
    base.update(kw)
    return cov.OpCoverage(**base)


def test_score_standard_op():
    # standard op, multiplatform + a domain ref, no large tier, econ n/a
    oc = _oc(domain_ref='ants.x', economic_verdict='n/a')
    # axes: platform(ok) + reference(ok) -> 2/2
    assert cov.score(oc) == (2, 2)


def test_score_marquee_full_vs_unmet():
    full = _oc(tier='marquee', domain_ref='ants.registration',
               domain_ref_realism='real_planted', input_realism='real_planted',
               economic_verdict='n/a')
    # platform, reference, real_input, domain_on_real -> 4/4
    assert cov.score(full) == (4, 4)
    unmet = _oc(tier='marquee', domain_ref='fsl.flameo',
                domain_ref_realism='synthetic', input_realism='synthetic',
                economic_verdict='n/a')
    sat, app = cov.score(unmet)
    assert app == 4 and sat == 2          # platform + reference only


def test_economic_is_not_in_the_completeness_score():
    # economic is an indicator/result, NOT a scored coverage requirement:
    # a not-multiplicative op must not lose a point for it.
    oc = _oc(coverage=cov.MULTIPLATFORM, ref_strength=cov.STRONG_REF,
             economic_verdict='not multiplicative enough',
             economic_authoritative=True)
    assert 'economic' not in [n for n, _ in cov.axes_status(oc)]
    assert cov.score(oc) == (2, 2)          # platform + reference only


def test_marquee_unmet_selector():
    full = _oc(qualname='a', tier='marquee', domain_ref='ants.x',
               domain_ref_realism='real_planted', input_realism='real_planted')
    unmet = _oc(qualname='b', tier='marquee', input_realism='synthetic')
    std = _oc(qualname='c', tier='standard', input_realism='synthetic')
    got = {r.qualname for r in cov._marquee_unmet([full, unmet, std])}
    assert got == {'b'}                   # full met; std not marquee


def test_build_carries_tier_and_orphans_json():
    cat = [{'qualname': 'pkg.winner', 'jit': 'pass'}]
    o2c = {'pkg.winner': _c('winner', tier='marquee')}
    rows = [_vrow('nitrix-jax', 'jax-cpu', 'ok', _REP, case='winner'),
            _vrow('nitrix-jax', 'jax-cuda12', 'ok', _REP, case='winner')]
    recs = cov.build_coverage(rows, cat, o2c)
    assert recs[0].tier == 'marquee'
    doc = cov.render_json(recs, orphans=[('pkg.ghost', 'marquee')])
    assert doc['summary']['orphan_cases'] == 1
    assert doc['orphan_cases'] == [
        {'qualname': 'pkg.ghost', 'tier': 'marquee'}]
    md = cov.render_markdown(recs, orphans=[('pkg.ghost', 'marquee')])
    assert 'absent from the catalogue' in md and 'MARQUEE' in md
