# -*- coding: utf-8 -*-
"""Tier-2 signal-processing cases (polynomial_detrend / tsconv /
lomb_scargle_periodogram).

CPU build + oracle agreement for the runnable (jax + numpy/scipy) baselines;
the cupy GPU ref is skipped here (needs a device + the refs env).  All three
are GPU-pure (lstsq / cross-correlation / trig sums -- no cuSolver), so the
cupy ref is skipped only for lack of a device, not on principle.
"""
import pytest

from nperf.cases import (
    lomb_scargle_periodogram,
    polynomial_detrend,
    tsconv,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

# (module, small build param, CPU reference baseline name)
_CASES = [
    (polynomial_detrend, {'c': 4, 'obs': 128, 'degree': 3, 'seed': 0},
     'numpy.lstsq_detrend'),
    (tsconv, {'obs': 256, 'k': 15, 'seed': 0}, 'scipy.signal.correlate'),
    (lomb_scargle_periodogram, {'obs': 256, 'seed': 0},
     'scipy.signal.lombscargle'),
]


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_baseline_shape(mod, param, refname):
    built = mod._build(param)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'  # cupy ref


@pytest.mark.parametrize('mod,param,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, param, refname):
    built = mod._build(param)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        args = built.inputs_for(framework_of(provider_id))
        fid = compare(fn(*args), built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualnames_match_nitrix():
    assert (polynomial_detrend.CASE.op_qualname
            == 'nitrix.signal.polynomial_detrend')
    assert tsconv.CASE.op_qualname == 'nitrix.signal.tsconv'
    assert (lomb_scargle_periodogram.CASE.op_qualname
            == 'nitrix.signal.lomb_scargle_periodogram')
