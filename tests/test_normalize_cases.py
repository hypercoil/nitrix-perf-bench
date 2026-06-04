# -*- coding: utf-8 -*-
"""Tier-2 numerics-normalize cases (zscore / psc / robust / intensity).

CPU build + oracle agreement for the runnable host baselines (jax + the
references: scipy.stats.zscore for the z-score -- the canonical domain tool --
and numpy reimplementations carrying nitrix's eps / 1.4826 constant for the
rest); the cupy GPU refs are skipped here (need a device + the refs env). The
zscore row doubles as the scipy.stats warranted-claim check.
"""
import numpy as np
import pytest

from nperf.cases import (
    intensity_normalize,
    psc_normalize,
    robust_zscore_normalize,
    zscore_normalize,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_MODS = [zscore_normalize, psc_normalize, robust_zscore_normalize,
         intensity_normalize]
_P = {'n': 128, 'seed': 0}
_REF = {
    'zscore_normalize': 'scipy.stats.zscore',
    'psc_normalize': 'numpy.psc',
    'robust_zscore_normalize': 'numpy.robust_zscore',
    'intensity_normalize': 'numpy.intensity',
}


@pytest.mark.parametrize('mod', _MODS)
def test_baselines(mod):
    names = set(mod._build(_P).baselines)
    assert {'nitrix-jax', _REF[mod.CASE.name]} <= names
    assert mod._build(_P).ratio_reference == 'nitrix-jax'
    gpu = [n for n in names if n.startswith('cupy.')]
    assert len(gpu) == 1
    assert requires_of(mod._build(_P).baselines[gpu[0]][0]) == 'gpu'


@pytest.mark.parametrize('mod', _MODS)
def test_host_baselines_match_oracle(mod):
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


def test_op_qualnames():
    assert (zscore_normalize.CASE.op_qualname
            == 'nitrix.numerics.zscore_normalize')
    assert psc_normalize.CASE.op_qualname == 'nitrix.numerics.psc_normalize'
    assert (robust_zscore_normalize.CASE.op_qualname
            == 'nitrix.numerics.robust_zscore_normalize')
    assert (intensity_normalize.CASE.op_qualname
            == 'nitrix.numerics.intensity_normalize')
