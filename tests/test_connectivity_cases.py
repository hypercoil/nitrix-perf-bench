# -*- coding: utf-8 -*-
"""Morphology connectivity cases: distance_transform_edt /
connected_components / largest_connected_component.

Pins the warranted comparison (EDT matches scipy's exact EDT; largest-CC
matches scipy's boolean co-oracle) and -- for connected_components, whose label
IDs are a permutation -- the *partition* match (the invariant) + the contracts.
"""
import numpy as np
import pytest

from nperf.cases import connected_components as cc_mod
from nperf.cases import distance_transform_edt as edt_mod
from nperf.cases import largest_connected_component as lcc_mod
from nperf.cases._connectivity import blob_mask, scipy_label
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_GATED = [edt_mod, lcc_mod]


@pytest.mark.parametrize('mod', [edt_mod, lcc_mod, cc_mod],
                         ids=lambda m: m.CASE.name)
def test_contract(mod):
    built = mod._build(mod.CASE.representative)
    assert 'nitrix-jax' in built.baselines
    assert built.ratio_reference == 'nitrix-jax'
    assert mod.CASE.op_qualname == f'nitrix.morphology.{mod.CASE.name}'
    assert mod.CASE.large_param_points
    assert max(p['d'] for p in mod.CASE.large_param_points) > (
        mod.CASE.representative['d'])
    assert mod.CASE.complexity


@pytest.mark.parametrize('mod', _GATED, ids=lambda m: m.CASE.name)
def test_gated_host_baselines_match_oracle(mod):
    case = mod.CASE
    for p in case.param_points:
        built = mod._build(p)
        for bname, (pid, fn) in built.baselines.items():
            if pid == 'cupy':
                continue
            out = np.asarray(fn(*built.inputs_for(framework_of(pid))),
                             np.float64)
            fid = compare(out, built.fp64_reference,
                          rtol=case.rtol, atol=case.atol)
            assert fid['status'] == 'pass', (
                f'{case.name}/{bname}@{p}: rel_to_tol={fid["rel_to_tol"]:.3g}')


def _same_partition(x, y):
    ax, ay = {}, {}
    for u, v in zip(x.tolist(), y.tolist()):
        if ax.get(u, v) != v or ay.get(v, u) != u:
            return False
        ax[u], ay[v] = v, u
    return True


def test_connected_components_partition_matches_scipy():
    # The label IDs differ (a permutation), but the partition -- which voxels
    # group together -- must match scipy on the foreground (task-level
    # correctness, the invariant the bench can't gate elementwise).
    import jax.numpy as jnp
    from nitrix.morphology import connected_components
    mask = blob_mask((40, 40, 40), seed=0)
    nx = np.asarray(connected_components(jnp.asarray(mask)))
    sp = scipy_label(mask)
    assert _same_partition(nx[mask], sp[mask])
    assert cc_mod._build(cc_mod.CASE.representative).fp64_reference is None
