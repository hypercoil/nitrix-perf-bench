# -*- coding: utf-8 -*-
"""Fidelity arithmetic (core/fidelity.py).

The headline is ``rel_to_tol`` and it must be **gate-consistent**
(``pass`` ⟺ ``rel_to_tol ≤ 1``) and **well-behaved on zero-centred outputs**,
where a bare element-wise ``max_rel`` would explode (SCHEMA_AND_LIFECYCLE §C).
"""
import numpy as np
import pytest

from nperf.core.fidelity import compare


def test_identical_is_pass_zero_error():
    x = np.linspace(-3, 3, 100).reshape(10, 10)
    fid = compare(x, x, rtol=1e-3, atol=1e-4)
    assert fid['status'] == 'pass'
    assert fid['rel_to_tol'] == 0.0
    assert fid['n_mismatched'] == 0
    assert fid['max_abs'] == 0.0


def test_gate_consistency_pass_just_under_tol():
    ref = np.ones((4, 4))
    atol, rtol = 1e-4, 1e-3
    tol = atol + rtol * 1.0
    out = ref + 0.99 * tol  # just inside tolerance everywhere
    fid = compare(out, ref, rtol=rtol, atol=atol)
    assert fid['status'] == 'pass'
    assert fid['rel_to_tol'] <= 1.0


def test_gate_consistency_fail_just_over_tol():
    ref = np.ones((4, 4))
    atol, rtol = 1e-4, 1e-3
    tol = atol + rtol * 1.0
    out = ref.copy()
    out[0, 0] = ref[0, 0] + 1.01 * tol  # one element just outside
    fid = compare(out, ref, rtol=rtol, atol=atol)
    assert fid['status'] == 'fail'
    assert fid['rel_to_tol'] > 1.0
    assert fid['n_mismatched'] == 1


def test_zero_centred_does_not_blow_up_rel_to_tol():
    # A near-zero reference element with a tiny absolute error: a *bare*
    # relative error would be enormous; rel_to_tol stays bounded because atol
    # floors the denominator, and the comparison correctly passes.
    ref = np.array([[0.0, 1.0], [2.0, -1.0]])
    out = ref + 1e-6  # well within atol=1e-4
    fid = compare(out, ref, rtol=1e-3, atol=1e-4)
    assert fid['status'] == 'pass'
    assert fid['rel_to_tol'] < 1.0
    # max_rel is guarded to significant |ref|, so the 0.0 element is excluded.
    assert np.isfinite(fid['max_rel'])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match='shape mismatch'):
        compare(np.zeros((3, 3)), np.zeros((3, 4)), rtol=1e-3, atol=1e-4)


def test_record_carries_threshold_and_oracle():
    fid = compare(np.zeros((2, 2)), np.zeros((2, 2)), rtol=2e-3, atol=5e-4)
    assert fid['threshold'] == {
        'rtol': 2e-3, 'atol': 5e-4, 'scope': 'per_case',
    }
    assert fid['oracle']['kind'] == 'fp64_full'
