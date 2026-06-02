# -*- coding: utf-8 -*-
"""Tier-2 lomb_scargle_interpolate case (Power-2014 joint-GLM interpolation).

This op has **no cross-impl fidelity oracle** (`fp64_reference=None`): the
censored-frame reconstruction is regularisation-sensitive (ill-conditioned
masked Gram). The well-defined correctness property is the **observed-sample
splice-through** -- the joint fit passes through observed samples exactly --
so that is what the test asserts (in place of the fidelity gate), for all
runnable (jax + numpy) baselines.
"""
import numpy as np

from nperf.cases import lomb_scargle_interpolate as lsi
from nperf.providers import framework_of, requires_of

_P = {'V': 16, 'obs': 256, 'seed': 0}


def test_no_cross_impl_oracle_with_note():
    built = lsi._build(_P)
    assert built.fp64_reference is None  # ill-posed censored-frame target
    assert built.fidelity_note and 'splice-through' in built.fidelity_note
    assert built.ratio_reference == 'nitrix-jax'


def test_baselines_and_gpu_ref():
    built = lsi._build(_P)
    assert {'nitrix-jax', 'numpy.joint_glm'} <= set(built.baselines)
    # the cupy joint-GLM is the GPU-only ref (fails at K>=256 on this stack).
    assert requires_of(built.baselines['cupy.joint_glm'][0]) == 'gpu'


def test_splice_through_is_exact():
    '''The load-bearing correctness property: observed frames pass through
    unchanged for every host-runnable baseline.'''
    built = lsi._build(_P)
    data, mask = built.inputs_for('numpy')
    data = np.asarray(data)
    mask = np.asarray(mask)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy ref: needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        err = np.max(np.abs(out[:, mask] - data[:, mask]))
        assert err < 1e-5, f'{name}: splice-through err {err:.2e}'


def test_op_qualname_matches_nitrix():
    assert (lsi.CASE.op_qualname
            == 'nitrix.signal.lomb_scargle_interpolate')
