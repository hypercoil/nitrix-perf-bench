# -*- coding: utf-8 -*-
"""Tier-2 resample case (nitrix.geometry.resample vs ANTsPy / scipy).

Linear image resize with align_corners=True. All samples are in-bounds, so it
has a clean fp64 oracle (no boundary divergence). The ANTsPy reference
(``resample_image``, interp_type=0) matches nitrix exactly but runs in its own
refs env, so the in-unit-env test exercises the jax + scipy baselines; the ANTs
+ cupy refs are checked in the matrix run.
"""
import numpy as np

from nperf.cases import resample
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_P = {'shape': [48, 48], 'out': [96, 96], 'seed': 0}


def test_baselines_and_ants_provider():
    built = resample._build(_P)
    assert set(built.baselines) == {
        'nitrix-jax', 'ants.resample_image',
        'scipy.ndimage.map_coordinates',
        'cupyx.scipy.ndimage.map_coordinates'}
    assert built.ratio_reference == 'nitrix-jax'
    # ANTs is an isolated-env CPU-only reference (its refs env ships CPU jax,
    # so requires='cpu' -- runs on jax-cpu, like statsmodels).
    ants_prov = built.baselines['ants.resample_image'][0]
    assert framework_of(ants_prov) == 'ants'
    assert requires_of(ants_prov) == 'cpu'
    assert requires_of(
        built.baselines['cupyx.scipy.ndimage.map_coordinates'][0]) == 'gpu'


def test_host_baselines_match_oracle():
    '''nitrix + scipy match the fp64 oracle (ANTs/cupy need other envs;
    ANTs parity is verified in the matrix -- it matched nitrix to 0.0).'''
    built = resample._build(_P)
    for name, (provider_id, fn) in built.baselines.items():
        if framework_of(provider_id) not in ('jax', 'numpy'):
            continue  # ants (own env) + cupy (gpu): not in the unit env
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference,
                      rtol=resample.CASE.rtol, atol=resample.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_op_qualname():
    assert resample.CASE.op_qualname == 'nitrix.geometry.resample'
