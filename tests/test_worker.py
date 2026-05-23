# -*- coding: utf-8 -*-
"""Subprocess worker + orchestrator failure handling.

The worker runs one attempt in its own process (honest per-attempt memory); the
orchestrator turns a worker that dies without a row into a classified failure
row so the sweep never aborts.  CPU-only (conftest pins JAX_PLATFORMS=cpu).
"""
import subprocess

from nperf.core import Status
from nperf.run import _spawn_worker, _synthesize_failure

# A tiny dense_matmul point keeps the spawn cheap.
_SPEC = {
    'run_id': 'test', 'case': 'dense_matmul',
    'param_point': {'m': 16, 'k': 8, 'n': 12, 'seed': 0},
    'warmup': 1, 'repeats': 2,
}


def test_worker_roundtrip_emits_isolated_row():
    spec = {**_SPEC, 'baseline': 'numpy-matmul', 'platform': 'jax-cpu'}
    rec = _spawn_worker(spec, 'jax-cpu', framework='numpy', timeout=120)
    assert rec.status == Status.OK
    assert rec.baseline == 'numpy-matmul'
    assert rec.metrics is not None and 'steady_time' in rec.metrics
    # Provenance was captured *in the worker* and marked subprocess-isolated.
    assert rec.provenance.get('measurement_isolation') == 'subprocess'


def test_orchestrator_synthesizes_failure_for_dead_worker():
    # An unknown baseline makes the worker raise (KeyError) and exit non-zero
    # without writing a row -> the orchestrator must synthesize a failure row,
    # not crash, so the rest of the sweep can proceed.
    spec = {**_SPEC, 'baseline': 'does-not-exist', 'platform': 'jax-cpu'}
    rec = _spawn_worker(spec, 'jax-cpu', framework='jax', timeout=120)
    assert rec.status != Status.OK
    assert rec.failure_detail is not None
    assert rec.failure_detail.get('returncode', 0) != 0


def test_synthesize_failure_classifies_sigkill_as_oom():
    proc = subprocess.CompletedProcess(
        args=[], returncode=-9, stdout='', stderr='Killed',
    )
    base = dict(
        run_id='r', case='c', param_point={}, baseline='b',
        platform='jax-cuda12', framework='jax', provenance={},
    )
    rec = _synthesize_failure(base, proc)
    assert rec.status == Status.OOM
    assert rec.failure_detail['signal'] == 9
