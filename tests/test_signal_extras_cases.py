# -*- coding: utf-8 -*-
"""Signal-extras family (nitrix.stats fourier): instantaneous phase/frequency,
the fused env_inst readout, and the frequency-domain product filters.

Pins the contract + that nitrix matches its fp64 oracle (scipy hilbert-based
for the phase/freq/env tracks; an exact numpy np.fft reimpl for the product
filters), which doubles as cross-tool agreement with the community impl.
"""
import jax
import numpy as np
import pytest

from nperf.measure import load_case

_EXPECT = {
    'instantaneous_phase': {'nitrix-jax', 'scipy.instantaneous_phase',
                            'cupy.instantaneous_phase'},
    'instantaneous_frequency': {'nitrix-jax', 'scipy.instantaneous_frequency',
                                'cupy.instantaneous_frequency'},
    'env_inst': {'nitrix-jax', 'scipy.env_inst', 'cupy.env_inst'},
    'product_filter': {'nitrix-jax', 'numpy.product_filter',
                       'cupy.product_filter'},
    'product_filtfilt': {'nitrix-jax', 'numpy.product_filtfilt',
                         'cupy.product_filtfilt'},
}


@pytest.mark.parametrize('name', list(_EXPECT))
def test_contract_and_oracle(name):
    c = load_case(name)
    bp = c.build({'n_sig': 16, 't': 512, 'seed': 0})
    assert set(bp.baselines) == _EXPECT[name]
    assert bp.ratio_reference == 'nitrix-jax'
    assert c.op_qualname.startswith('nitrix.signal.')
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    ref = np.asarray(bp.fp64_reference)
    assert out.shape == ref.shape, f'{name}: {out.shape} vs {ref.shape}'
    rel = np.max(np.abs(out - ref) / (c.atol + c.rtol * np.abs(ref)))
    assert rel <= 1.0, f'{name}: rel_to_tol {rel:.2f} > 1 (vs oracle)'


def test_inst_freq_is_one_shorter():
    '''instantaneous_frequency is a discrete derivative -> the time axis is one
    sample shorter than the input (the convention nitrix + scipy share).'''
    c = load_case('instantaneous_frequency')
    bp = c.build({'n_sig': 8, 't': 256, 'seed': 0})
    out = np.asarray(jax.block_until_ready(
        bp.baselines['nitrix-jax'][1](*bp.inputs_for('jax'))))
    assert out.shape == (8, 255)
