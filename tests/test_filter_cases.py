# -*- coding: utf-8 -*-
"""Tier-2 IIR signal-filter cases (sosfilt / sosfiltfilt), B18 Win 2.

The headline ``nitrix-jax`` row is the **default** call (``backend='auto'`` ->
fft on GPU, scan on CPU) -- the engine users actually hit, which the old
scan-pinned case never measured.  The ``fft`` / ``scan`` / ``associative``
engines run as labelled variants and must all agree with the scipy fp64 oracle
(the load-bearing correctness check, especially for the FFT engine).

The FFT win is only honest where the FFT path actually applies: a near-unstable
filter whose impulse does not decay within 2**15 taps falls back to the
recurrence (with a warning).  ``test_fft_fallback_warns_and_stays_correct``
pins that -- so a bench cannot claim the FFT win on a filter that silently fell
back.  The cupy GPU refs are skipped here (need a device + the refs env).
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from nitrix.signal import sosfilt as nx_sosfilt

from nperf.cases import sosfilt, sosfiltfilt
from nperf.cases._filters import scipy_sosfilt, sharp_sos
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'channels': 8, 'obs': 1024, 'seed': 0}


def test_sosfilt_baselines():
    built = sosfilt._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'nitrix-jax-fft', 'nitrix-jax-scan', 'nitrix-jax-assoc',
        'scipy.signal.sosfilt', 'cupyx.scipy.signal.sosfilt'}
    assert built.ratio_reference == 'nitrix-jax'
    assert requires_of(built.baselines['cupyx.scipy.signal.sosfilt'][0]) \
        == 'gpu'


def test_sosfiltfilt_baselines():
    built = sosfiltfilt._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'nitrix-jax-fft', 'nitrix-jax-scan',
        'scipy.signal.sosfiltfilt', 'cupyx.scipy.signal.sosfiltfilt'}
    assert built.ratio_reference == 'nitrix-jax'


def test_all_engines_match_oracle():
    '''Every nitrix engine (auto / fft / scan / associative) matches the scipy
    fp64 oracle -- the correctness gate for the FFT-convolution engine that is
    now the GPU default.'''
    for mod in (sosfilt, sosfiltfilt):
        built = mod._build(_P)
        for name, (provider_id, fn) in built.baselines.items():
            if requires_of(provider_id) == 'gpu':
                continue  # cupy ref: needs a device + the refs env
            out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
            fid = compare(out, built.fp64_reference,
                          rtol=mod.CASE.rtol, atol=mod.CASE.atol)
            assert fid['status'] == 'pass', (
                f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


def test_fft_fallback_warns_and_stays_correct():
    '''A near-unstable filter (impulse not decayed within 2**15 taps) must make
    backend='fft' fall back to the recurrence -- WITH a warning -- and still
    match scipy.  This is the honesty guard: the FFT win cannot be claimed on a
    filter that silently fell back.'''
    sos = sharp_sos()
    x = np.random.default_rng(0).standard_normal((4, 8192)).astype(np.float32)
    ref = scipy_sosfilt(sos.astype(np.float64))(x.astype(np.float64))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = np.asarray(jax.block_until_ready(
            nx_sosfilt(jnp.asarray(x), sos, backend='fft')))
    assert any('impulse' in str(w.message).lower() for w in caught), (
        'expected an impulse-not-decayed fallback warning'
    )
    # fell back, but still correct (the recurrence is exact)
    assert np.abs(out - ref).max() < 1e-2


def test_op_qualnames():
    assert sosfilt.CASE.op_qualname == 'nitrix.signal.sosfilt'
    assert sosfiltfilt.CASE.op_qualname == 'nitrix.signal.sosfiltfilt'
