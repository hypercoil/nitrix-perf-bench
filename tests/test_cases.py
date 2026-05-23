# -*- coding: utf-8 -*-
"""The first real case (cases/semiring_matmul.py).

Checks the case *builds* correctly on CPU: the baseline registry is shaped as
expected, the fp64 oracle is real double precision, and every runnable
*jax* strategy (the JAX reference and the naive-dense form) agrees with the
oracle.  The Pallas baseline is registered but not executed here — it needs
CUDA (covered by the runner on a GPU host, recorded as a ``skipped`` row
off-GPU).  The ``torch-dense`` baseline lives in its own (torch) env, so its
math is checked in ``test_refs_torch.py`` under ``importorskip``, not here.
"""
import numpy as np
import pytest

from nperf.cases import semiring_matmul
from nperf.core.fidelity import compare

_SMALL = {'m': 32, 'k': 16, 'n': 24, 'seed': 0}


def _point(algebra: str) -> dict:
    return {**_SMALL, 'algebra': algebra}


def test_baseline_registry_shape():
    built = semiring_matmul._build(_point('log'))
    assert set(built.baselines) == {'nitrix-jax', 'nitrix-pallas',
                                    'naive-dense', 'torch-dense'}
    assert built.ratio_reference == 'nitrix-jax'
    # jnp-matmul ceiling is added only for the real semiring.
    real_built = semiring_matmul._build(_point('real'))
    assert 'jnp-matmul' in real_built.baselines


def test_oracle_is_fp64_and_correct_shape():
    built = semiring_matmul._build(_point('log'))
    assert built.fp64_reference.dtype == np.float64
    assert built.fp64_reference.shape == (_SMALL['m'], _SMALL['n'])


@pytest.mark.parametrize('algebra',
                         ['real', 'log', 'tropical_max_plus', 'euclidean'])
def test_runnable_baselines_match_oracle(algebra):
    built = semiring_matmul._build(_point(algebra))
    args = built.inputs_for('jax')
    for name, (framework, fn) in built.baselines.items():
        if framework != 'jax':
            continue  # torch-dense runs in its own env (test_refs_torch.py)
        if name == 'nitrix-pallas':
            continue  # needs CUDA; exercised by the runner on a GPU host
        out = np.asarray(fn(*args), dtype=np.float64)
        fid = compare(out, built.fp64_reference, rtol=1e-3, atol=1e-4)
        assert fid['status'] == 'pass', (
            f'{algebra}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}'
        )
