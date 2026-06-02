# -*- coding: utf-8 -*-
"""Tier-2 tangent-space SPD case (tangent_project_spd).

CPU build + oracle agreement against the fp64 eigh-based log map. The host
baselines -- nitrix (jax, CPU here) and **nilearn**'s tangent kernel -- must
both match the oracle; nilearn is the first domain-tool reference in the suite
and the canonical neuroimaging tangent-space implementation, so its agreement
with nitrix is the load-bearing check (the right-target match: same affine-
invariant ``log(R^-1/2 X R^-1/2)``, same reference R).
"""
from nperf.cases import tangent_project_spd as tps
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'b': 8, 'd': 32, 'seed': 0}


def test_baselines_and_gpu_ref():
    built = tps._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'nilearn.tangent', 'cupy.eigh_tangent'}
    assert built.ratio_reference == 'nitrix-jax'
    # cupy is the GPU-only reference; nilearn is the CPU floor.
    assert requires_of(built.baselines['cupy.eigh_tangent'][0]) == 'gpu'
    assert requires_of(built.baselines['nilearn.tangent'][0]) is None


def test_host_baselines_match_oracle():
    built = tps._build(_P)
    c = tps.CASE
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy ref: needs a device + the refs env
        out = fn(*built.inputs_for(framework_of(provider_id)))
        fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
        assert fid['status'] == 'pass', (
            f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_nitrix_matches_nilearn():
    '''The right-target check: nitrix's affine-invariant tangent equals
    nilearn's connectome tangent kernel given the same reference R.'''
    import numpy as np

    built = tps._build(_P)
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax')))
    nil = np.asarray(built.baselines['nilearn.tangent'][1](
        *built.inputs_for('numpy')))
    assert np.max(np.abs(nit - nil)) < 5e-3


def test_oracle_shape_and_symmetry():
    built = tps._build(_P)
    ref = built.fp64_reference  # (b, d, d), each a symmetric tangent matrix
    assert ref.shape == (_P['b'], _P['d'], _P['d'])
    sym_err = abs(ref - ref.transpose(0, 2, 1)).max()
    assert sym_err < 1e-9


def test_op_qualname_matches_nitrix():
    assert (tps.CASE.op_qualname
            == 'nitrix.linalg.tangent_project_spd')
