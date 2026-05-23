# -*- coding: utf-8 -*-
"""P2 cross-framework (torch) wiring — all CPU-only / torch-optional.

What's checked here without needing torch installed in the test env: the
provider-aware worker-interpreter resolution, that a jax-only worker can build
the semiring_matmul point *without* importing torch (laziness), that a missing
torch env classifies as a clean ``env_failed`` (not a scary compile_error), and
the host-conversion hook for torch outputs.  The actual torch math is checked
under ``importorskip`` so it runs anywhere torch exists (e.g. the refs env) and
skips on the jax-only unit env.
"""
import sys

import numpy as np
import pytest

from nperf.cases import semiring_matmul
from nperf.core.sync import SYNC
from nperf.measure import _host_f64, classify_message
from nperf.run import _worker_python


# ---- worker interpreter resolution (framework + platform) --------------- #
def test_jax_resolution_unchanged_by_platform_var(monkeypatch):
    # jax/numpy keep the historical platform-only resolution.
    monkeypatch.setenv('NPERF_PYTHON_JAX_CPU', '/envs/jaxcpu/python')
    monkeypatch.delenv('NPERF_PYTHON_TORCH', raising=False)
    assert _worker_python('jax-cpu', 'jax') == '/envs/jaxcpu/python'
    assert _worker_python('jax-cpu', 'numpy') == '/envs/jaxcpu/python'


def test_torch_framework_env_wins_over_platform(monkeypatch):
    # A torch attempt needs its own interpreter even on a jax platform: the
    # framework override beats the platform-wide one.
    monkeypatch.setenv('NPERF_PYTHON_JAX_CPU', '/envs/jaxcpu/python')
    monkeypatch.setenv('NPERF_PYTHON_TORCH', '/envs/torch/python')
    assert _worker_python('jax-cpu', 'torch') == '/envs/torch/python'
    # ...but jax on that platform still uses the platform interpreter.
    assert _worker_python('jax-cpu', 'jax') == '/envs/jaxcpu/python'


def test_torch_platform_specific_env_is_most_specific(monkeypatch):
    monkeypatch.setenv('NPERF_PYTHON_TORCH', '/envs/torch/python')
    monkeypatch.setenv('NPERF_PYTHON_TORCH_JAX_CUDA12', '/envs/tg/python')
    assert _worker_python('jax-cuda12', 'torch') == '/envs/tg/python'


def test_resolution_falls_back_to_this_interpreter(monkeypatch):
    for k in ('NPERF_PYTHON_TORCH', 'NPERF_PYTHON_JAX_CPU',
              'NPERF_WORKER_PYTHON'):
        monkeypatch.delenv(k, raising=False)
    assert _worker_python('jax-cpu', 'torch') == sys.executable


# ---- case wiring: torch baseline present but lazy ----------------------- #
def test_case_registers_torch_dense_provider():
    built = semiring_matmul.CASE.build(
        {'m': 8, 'k': 8, 'n': 8, 'algebra': 'log', 'seed': 0}
    )
    assert built.baselines['torch-dense'][0] == 'torch'


def test_build_does_not_import_torch():
    # Building the point (for any baseline) must not import torch -- that is
    # what lets a jax-only worker run its own baselines without the refs env.
    sys.modules.pop('torch', None)
    semiring_matmul.CASE.build(
        {'m': 8, 'k': 8, 'n': 8, 'algebra': 'real', 'seed': 0}
    )
    assert 'torch' not in sys.modules


# ---- graceful degradation: missing refs env ----------------------------- #
def test_missing_torch_classifies_env_failed():
    status, detail = classify_message("ModuleNotFoundError: No module "
                                      "named 'torch'")
    assert status.value == 'env_failed'
    assert detail['reason'] == 'provider_env_missing'


# ---- host conversion + sync hook ---------------------------------------- #
def test_host_f64_jax_and_torch_paths():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    out = _host_f64(arr, 'jax')
    assert out.dtype == np.float64 and out.shape == (1, 2)

    class _FakeTensor:  # mimics the torch detach().cpu().numpy() chain
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return arr

    out_t = _host_f64(_FakeTensor(), 'torch')
    assert out_t.dtype == np.float64
    np.testing.assert_allclose(out_t, arr)


def test_torch_sync_registered():
    assert 'torch' in SYNC


# ---- torch math, where torch exists (refs env); skipped on jax-only ----- #
def test_torch_semiring_matches_oracle():
    torch = pytest.importorskip('torch')
    param = {'m': 16, 'k': 24, 'n': 12, 'algebra': 'log', 'seed': 1}
    built = semiring_matmul.CASE.build(param)
    a, b = built.inputs_for('torch')
    out = semiring_matmul._torch_semiring(torch, 'log', a, b)
    got = out.detach().cpu().numpy().astype(np.float64)
    # vs the fp64 oracle the case built (the nitrix reference, in double).
    np.testing.assert_allclose(got, built.fp64_reference, rtol=1e-3, atol=1e-4)
