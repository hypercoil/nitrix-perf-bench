# -*- coding: utf-8 -*-
"""Tier-2 Hilbert / analytic-signal cases (analytic_signal / hilbert_transform
/ envelope).

CPU build + oracle agreement for the runnable (jax + scipy) baselines; the cupy
GPU ref is skipped here (needs a device + the refs env).  ``analytic_signal``
is complex-valued, so this also exercises the complex fidelity path.
"""
import numpy as np
import pytest

from nperf.cases import analytic_signal, envelope, hilbert_transform
from nperf.core.fidelity import compare
from nperf.providers import requires_of

_P = {'n_sig': 8, 't': 128, 'seed': 0}
_CASES = [
    (analytic_signal, 'scipy.signal.hilbert'),
    (hilbert_transform, 'scipy.signal.hilbert'),
    (envelope, 'scipy.signal.hilbert'),
]


@pytest.mark.parametrize('mod,refname', _CASES)
def test_baseline_shape(mod, refname):
    built = mod._build(_P)
    names = set(built.baselines)
    assert {'nitrix-jax', refname} <= names
    assert built.ratio_reference == 'nitrix-jax'
    for extra in names - {'nitrix-jax', refname}:
        assert requires_of(built.baselines[extra][0]) == 'gpu'  # cupy ref


@pytest.mark.parametrize('mod,refname', _CASES)
def test_cpu_baselines_match_oracle(mod, refname):
    built = mod._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy GPU ref: needs a device + the refs env
        fid = compare(fn(*built.inputs_for(provider_id)),
                      built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_analytic_signal_oracle_is_complex():
    built = analytic_signal._build(_P)
    assert np.iscomplexobj(built.fp64_reference)


def test_op_qualnames_match_nitrix():
    assert analytic_signal.CASE.op_qualname == 'nitrix.signal.analytic_signal'
    assert (hilbert_transform.CASE.op_qualname
            == 'nitrix.signal.hilbert_transform')
    assert envelope.CASE.op_qualname == 'nitrix.signal.envelope'
