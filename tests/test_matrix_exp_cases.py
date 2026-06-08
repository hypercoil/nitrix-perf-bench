# -*- coding: utf-8 -*-
"""Registration primitive: ``nitrix.linalg.matrix_exp`` (affine bottleneck).

CPU build + oracle agreement for the host baselines: nitrix's
scaling-and-squaring Taylor expm vs ``scipy.linalg.expm`` (the fp64 oracle +
CPU floor) and ``jax.scipy.linalg.expm`` (the Padé twin, on CPU here).  The
warranted-comparison check: at a pinned spectral norm nitrix matches Padé to
fp32 round-off (the gate is ``atol=5e-4`` for that honest fp32 chain).
"""
import numpy as np

from nperf.cases import matrix_exp
from nperf.core.fidelity import compare
from nperf.providers import framework_of

_REFS = {'scipy.linalg.expm', 'jax.scipy.linalg.expm'}


def test_baselines():
    built = matrix_exp._build({'n': 16, 'seed': 0})
    names = set(built.baselines)
    assert {'nitrix-jax'} | _REFS <= names
    assert built.ratio_reference == 'nitrix-jax'
    # scipy is the fp64 oracle source (numpy framework); the GPU twin is jax.
    assert framework_of(built.baselines['scipy.linalg.expm'][0]) == 'numpy'
    assert framework_of(built.baselines['jax.scipy.linalg.expm'][0]) == 'jax'


def test_host_baselines_match_oracle():
    # All three baselines are host-runnable (jax on CPU); each must match the
    # fp64 scipy oracle within the (fp32-honest) gate.
    case = matrix_exp.CASE
    for p in case.param_points:
        built = matrix_exp._build(p)
        for name, (provider_id, fn) in built.baselines.items():
            out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
            fid = compare(out, built.fp64_reference,
                          rtol=case.rtol, atol=case.atol)
            assert fid['status'] == 'pass', (
                f"matrix_exp/{name} n={p['n']}: "
                f'rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


def test_op_qualname_and_scale_tier():
    case = matrix_exp.CASE
    assert case.op_qualname == 'nitrix.linalg.matrix_exp'
    # ships with a scalability case (COVERAGE_MANDATE §7-D): a larger-n tier
    # past the representative, plus the stated cost law.
    assert case.large_param_points
    big = max(p['n'] for p in case.large_param_points)
    assert big > case.representative['n']
    assert case.complexity and 'O(n^3)' in case.complexity
