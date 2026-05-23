# -*- coding: utf-8 -*-
"""The sparse message-passing case (cases/ell_edge_aggregate.py).

Checks it builds on CPU: the baseline registry (nitrix-jax + pyg), the fp64
oracle, and that the nitrix-jax reference matches the oracle for both supported
algebras (sum / max).  The PyG baseline runs in its own (torch) env, so its
math is checked under ``importorskip`` -- it runs where torch_geometric exists
(the refs env) and skips on the jax-only unit env.
"""
import numpy as np
import pytest

from nperf.cases import ell_edge_aggregate
from nperf.core.fidelity import compare

_SMALL = {'n': 64, 'k': 5, 'd_in': 8, 'd_out': 6, 'seed': 0}


def _point(algebra):
    return {**_SMALL, 'algebra': algebra}


def test_baseline_registry_shape():
    built = ell_edge_aggregate._build(_point('real'))
    assert set(built.baselines) == {'nitrix-jax', 'pyg'}
    assert built.baselines['pyg'][0] == 'torch'
    assert built.ratio_reference == 'nitrix-jax'


def test_oracle_shape_and_dtype():
    built = ell_edge_aggregate._build(_point('real'))
    assert built.fp64_reference.dtype == np.float64
    assert built.fp64_reference.shape == (_SMALL['n'], _SMALL['d_out'])


def test_build_does_not_import_torch():
    import sys

    sys.modules.pop('torch', None)
    ell_edge_aggregate._build(_point('tropical_max_plus'))
    assert 'torch' not in sys.modules


@pytest.mark.parametrize('algebra', ['real', 'tropical_max_plus'])
def test_nitrix_jax_matches_oracle(algebra):
    built = ell_edge_aggregate._build(_point(algebra))
    args = built.inputs_for('jax')
    fn = built.baselines['nitrix-jax'][1]
    out = np.asarray(fn(*args), dtype=np.float64)
    fid = compare(out, built.fp64_reference, rtol=1e-3, atol=1e-4)
    assert fid['status'] == 'pass', f'{algebra}: {fid["rel_to_tol"]:.3g}'


@pytest.mark.parametrize('algebra', ['real', 'tropical_max_plus'])
def test_pyg_matches_oracle(algebra):
    pytest.importorskip('torch_geometric')
    built = ell_edge_aggregate._build(_point(algebra))
    args = built.inputs_for('torch')
    out = built.baselines['pyg'][1](*args)
    got = out.detach().cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(got, built.fp64_reference, rtol=1e-3, atol=1e-4)
