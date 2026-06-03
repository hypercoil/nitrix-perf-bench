# -*- coding: utf-8 -*-
"""Tier-2 IIR signal-filter cases (sosfilt / sosfiltfilt).

CPU build + oracle agreement for the runnable (jax + scipy) baselines; the cupy
GPU ref is skipped here (needs a device + the refs env). sosfilt exposes two
recurrence backends (sequential scan + parallel associative-scan) -- both are
exercised and must agree with the scipy oracle, which is the load-bearing
correctness check for the parallel-prefix engine.
"""
import numpy as np

from nperf.cases import sosfilt, sosfiltfilt
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'channels': 8, 'obs': 512, 'seed': 0}


def test_sosfilt_baselines():
    built = sosfilt._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'nitrix-jax-assoc',
        'scipy.signal.sosfilt', 'cupyx.scipy.signal.sosfilt'}
    assert built.ratio_reference == 'nitrix-jax'
    assert requires_of(built.baselines['cupyx.scipy.signal.sosfilt'][0]) \
        == 'gpu'


def test_both_backends_match_oracle():
    '''Both nitrix recurrence engines (scan + associative) match the scipy
    fp64 oracle -- the correctness gate for the parallel-prefix engine.'''
    built = sosfilt._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy ref: needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference,
                      rtol=sosfilt.CASE.rtol, atol=sosfilt.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_sosfiltfilt_matches_oracle():
    built = sosfiltfilt._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference,
                      rtol=sosfiltfilt.CASE.rtol, atol=sosfiltfilt.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames():
    assert sosfilt.CASE.op_qualname == 'nitrix.signal.sosfilt'
    assert sosfiltfilt.CASE.op_qualname == 'nitrix.signal.sosfiltfilt'
