# -*- coding: utf-8 -*-
"""PERF_AUDIT divergent scipy.ndimage ports (B11 slice 3).

Two resolve to clean oracles -- distance_transform (exact EDT with a 1-voxel
tolerance, since nitrix is a quasi-Euclidean approximation) and
spatial_transform (in-bounds deformation, so boundary handling can't diverge).
median_filter has no shared oracle (boundary policies differ by design) and
exercises the no-cross-impl-oracle path: OK status, inconclusive fidelity, but
a perf ratio is still produced.
"""
import numpy as np
import pytest

from nperf import measure
from nperf.cases import distance_transform, median_filter, spatial_transform
from nperf.core import Status
from nperf.core.fidelity import compare
from nperf.providers import requires_of


@pytest.mark.parametrize('mod,param', [
    (distance_transform, {'shape': [24, 24], 'seed': 0}),
    (spatial_transform, {'shape': [24, 24], 'seed': 0}),
])
def test_clean_ops_match_oracle(mod, param):
    built = mod._build(param)
    assert built.fp64_reference is not None
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # GPU-only ref (cupy): needs a device + the refs env
        out = np.asarray(fn(*built.inputs_for(provider_id)), dtype=np.float64)
        fid = compare(out, built.fp64_reference,
                      rtol=mod.CASE.rtol, atol=mod.CASE.atol)
        assert fid['status'] == 'pass', (
            f'{mod.CASE.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )


def test_median_filter_declares_no_oracle():
    built = median_filter._build({'shape': [24, 24], 'size': 3, 'seed': 0})
    assert built.fp64_reference is None
    assert built.fidelity_note  # the recorded reason


def test_no_oracle_path_is_ok_inconclusive_with_ratio():
    case = measure.CASES['median_filter']
    param = {'shape': [24, 24], 'size': 3, 'seed': 0}
    built = case.build(param)
    # the GPU-only cupy ref isn't runnable in the unit env; the runner gates
    # it by platform (platform_not_applicable), but measure_attempt is called
    # directly here, so filter it out.
    cpu_baselines = [
        n for n in built.baselines
        if requires_of(built.baselines[n][0]) != 'gpu'
    ]
    recs = [
        measure.measure_attempt(
            case, param, built, name, platform='jax-cpu', run_id='t',
            prov={}, warmup=1, repeats=2)
        for name in cpu_baselines
    ]
    for r in recs:
        assert r.status == Status.OK
        assert r.fidelity['status'] == 'inconclusive'
        assert r.fidelity['reason']
        assert r.metrics['steady_time']['min'] >= 0  # measured
    # perf ratio is still produced for the non-reference baseline.
    measure.attach_ratios(recs, built.ratio_reference)
    others = [r for r in recs if r.baseline != built.ratio_reference]
    assert others and all(r.ratio is not None for r in others)


def test_op_qualnames_match_nitrix():
    assert (distance_transform.CASE.op_qualname
            == 'nitrix.morphology.distance_transform')
    assert (spatial_transform.CASE.op_qualname
            == 'nitrix.geometry.spatial_transform')
    assert (median_filter.CASE.op_qualname
            == 'nitrix.morphology.median_filter')
