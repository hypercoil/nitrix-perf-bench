# -*- coding: utf-8 -*-
"""Baseline selection: ``--baselines`` (allowlist) / ``--skip-baselines``.

The load-bearing property (DESIGN §1, "omission is data"): a skipped baseline
is **recorded as a ``skipped`` row**, never silently dropped -- so deliberately
skipping a pathological baseline (e.g. ``naive-dense``'s ~10-min cold compile)
leaves a visible, explained gap in the store without paying to measure it, and
historical comparisons are not corrupted by a vanished row.
"""
from dataclasses import replace

from nperf.core import Status
from nperf.measure import CASES
from nperf.providers import requires_of
from nperf.run import _select_baselines, run_case_inprocess

_NAMES = ['nitrix-jax', 'nitrix-pallas', 'naive-dense', 'torch-dense']


def test_select_baselines_allow_and_skip():
    # default: everything, order preserved.
    assert _select_baselines(_NAMES, None, frozenset()) == _NAMES
    # allowlist keeps only the named, in original order.
    assert _select_baselines(
        _NAMES, frozenset({'naive-dense', 'nitrix-jax'}), frozenset()
    ) == ['nitrix-jax', 'naive-dense']
    # denylist removes the named.
    assert _select_baselines(_NAMES, None, frozenset({'naive-dense'})) == [
        'nitrix-jax', 'nitrix-pallas', 'torch-dense'
    ]
    # skip wins over allow on overlap.
    assert _select_baselines(
        _NAMES, frozenset({'nitrix-jax', 'naive-dense'}),
        frozenset({'naive-dense'}),
    ) == ['nitrix-jax']


def _tiny_cov():
    '''A trivially-sized cov point so the in-process measurement is fast.'''
    return replace(CASES['cov'], param_points=[{'c': 8, 'n_obs': 16}])


def test_skip_records_a_skipped_row_not_a_drop():
    recs = run_case_inprocess(
        _tiny_cov(), platform='jax-cpu', warmup=1, repeats=2, prov={},
        run_id='t', skip=frozenset({'numpy.cov'}),
    )
    by = {r.baseline: r for r in recs}
    assert {'nitrix-jax', 'numpy.cov'} <= set(by)  # present, not dropped
    assert by['nitrix-jax'].status == Status.OK
    assert by['nitrix-jax'].metrics is not None
    skipped = by['numpy.cov']
    assert skipped.status == Status.SKIPPED
    assert skipped.metrics is None  # no measurement was taken
    assert skipped.failure_detail == {'reason': 'skipped_by_config'}


def test_allowlist_runs_only_selected():
    recs = run_case_inprocess(
        _tiny_cov(), platform='jax-cpu', warmup=1, repeats=2, prov={},
        run_id='t', allow=frozenset({'nitrix-jax'}),
    )
    by = {r.baseline: r.status for r in recs}
    assert by['nitrix-jax'] == Status.OK
    assert by['numpy.cov'] == Status.SKIPPED  # not in the allowlist


def test_cupy_provider_requires_gpu():
    assert requires_of('cupy') == 'gpu'   # GPU-only on-target ref
    assert requires_of('numpy') is None   # host ref, applicable anywhere


def test_gpu_only_ref_skipped_on_cpu_platform():
    # cov declares a cupy.cov GPU reference (provider requires gpu).  On the
    # cpu platform it must be recorded platform_not_applicable -- never run
    # (cupy ignores JAX_PLATFORMS, so running it would silently use the GPU and
    # mislabel the row jax-cpu).
    recs = run_case_inprocess(
        _tiny_cov(), platform='jax-cpu', warmup=1, repeats=2, prov={},
        run_id='t',
    )
    by = {r.baseline: r for r in recs}
    assert by['nitrix-jax'].status == Status.OK   # the cpu baselines still run
    assert by['cupy.cov'].status == Status.SKIPPED
    assert by['cupy.cov'].metrics is None
    assert by['cupy.cov'].failure_detail == {
        'reason': 'platform_not_applicable'
    }
