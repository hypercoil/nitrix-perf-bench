# -*- coding: utf-8 -*-
"""The metric + provider registries.

Metric registry = single source of truth for unit / direction / kind (+ the
fidelity gate threshold).  Provider registry = the cross-case framework + env
isolation a baseline runs under (baseline *names* stay case-local, so they are
deliberately NOT globally registered).  Both reject unknown entries clearly.
"""
import pytest

from nperf.cases import Case
from nperf.core import METRICS, Metric
from nperf.measure import _validate_case
from nperf.providers import Provider, framework_of, provider


# ---- metric registry ---------------------------------------------------- #
def test_metric_floor_present_with_directions():
    for name in ('steady_time', 'peak_hbm', 'fidelity'):
        assert name in METRICS
    assert METRICS['steady_time'].kind == 'distribution'
    assert METRICS['throughput'].direction == 'higher'
    # fidelity is catalogued (stored in the fidelity block) with its gate.
    fid = METRICS['fidelity']
    assert fid.kind == 'fidelity' and fid.threshold == 1.0
    assert all(m.direction in ('lower', 'higher') for m in METRICS.values())


def _case(metrics):
    return Case(
        name='t', output_independent=True, metrics=metrics,
        param_points=[{}], representative={}, build=lambda p: None,
    )


def test_validate_case_rejects_unknown_metric():
    with pytest.raises(ValueError, match='unknown metric'):
        _validate_case(_case(['steady_time', 'bogus_metric']))


def test_validate_case_accepts_known_metrics():
    assert _validate_case(_case(['steady_time', 'peak_hbm'])).name == 't'


# ---- provider registry -------------------------------------------------- #
def test_provider_lookup_and_framework():
    assert provider('jax').framework == 'jax'
    assert framework_of('numpy') == 'numpy'


def test_torch_provider_is_uv_isolated():
    # torch CPU wheels are uv-installable, so the cross-framework ref is a uv
    # env (a separate interpreter), not the pixi escape hatch.
    p = provider('torch')
    assert p.framework == 'torch' and p.isolation == 'uv'


def test_pyg_provider_is_uv_and_torch_framework():
    # Modern PyG message-passes on torch-native scatter_reduce -> pure-Python
    # uv install (no compiled torch-scatter), so it is NOT the pixi case; it
    # shares torch's framework (sync hook + host conversion).
    p = provider('pyg')
    assert p.isolation == 'uv'
    assert framework_of('pyg') == 'torch'


def test_unregistered_provider_raises_clearly():
    with pytest.raises(KeyError, match='unregistered provider'):
        framework_of('tensorflow')  # never registered


def test_pixi_provider_requires_reason():
    with pytest.raises(ValueError, match='pixi_reason'):
        Provider('torch', 'torch', isolation='pixi')
    # With a reason it is accepted (the audit trail, DESIGN §7).
    p = Provider('torch', 'torch', isolation='pixi',
                 pixi_reason='no PyPI wheel')
    assert p.isolation == 'pixi'


def test_bad_isolation_rejected():
    with pytest.raises(ValueError, match="'uv' or 'pixi'"):
        Provider('x', 'jax', isolation='conda')


def test_metric_is_frozen():
    m = Metric('x', 's', 'lower')
    with pytest.raises(Exception):
        m.unit = 'ms'  # frozen dataclass
