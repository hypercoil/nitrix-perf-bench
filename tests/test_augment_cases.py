# -*- coding: utf-8 -*-
"""Tier-2 augmentation cases (`nitrix.augment`).

CPU build + oracle agreement for the host-runnable baselines (jax + the numpy
reimplementation carrying nitrix's `eps`); the cupy GPU ref is skipped here
(needs a device + the refs-cupy env). The MONAI community baseline is gated on
`monai` being importable (it lives in the refs-monai env, not the base test
env) -- when present it doubles as the apples-to-apples community check.
"""
import numpy as np
import pytest

from nperf.cases import gamma_contrast
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of


def test_gamma_baselines():
    built = gamma_contrast._build(gamma_contrast.CASE.representative)
    assert set(built.baselines) == {
        'nitrix-jax', 'numpy.gamma_contrast', 'cupy.gamma_contrast',
        'monai.AdjustContrast'}
    assert built.ratio_reference == 'nitrix-jax'
    # cupy is the on-target GPU headline ref (GPU-only); MONAI the CPU
    # community baseline (CPU-only for now).
    assert requires_of(built.baselines['cupy.gamma_contrast'][0]) == 'gpu'
    assert requires_of(built.baselines['monai.AdjustContrast'][0]) == 'cpu'
    assert framework_of(built.baselines['monai.AdjustContrast'][0]) == 'monai'


def test_gamma_host_baselines_match_oracle():
    c = gamma_contrast.CASE
    built = gamma_contrast._build(c.representative)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy: needs a device + the refs-cupy env
        if provider_id == 'monai':
            continue  # covered by test_gamma_monai_agreement (gated)
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
        assert fid['status'] == 'pass', f'{name}: {fid["rel_to_tol"]:.3g}'


def test_gamma_monai_agreement():
    '''nitrix's gamma_contrast == MONAI AdjustContrast (the clean community
    baseline; ~7e-8 in fp32). Gated on monai (refs-monai env only).'''
    pytest.importorskip('monai')
    pytest.importorskip('torch')
    c = gamma_contrast.CASE
    built = gamma_contrast._build(c.representative)
    fn = built.baselines['monai.AdjustContrast'][1]
    out = np.asarray(fn(*built.inputs_for('monai')))
    fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
    assert fid['status'] == 'pass', f'monai: {fid["rel_to_tol"]:.3g}'


def test_gamma_op_qualname():
    assert gamma_contrast.CASE.op_qualname == 'nitrix.augment.gamma_contrast'
