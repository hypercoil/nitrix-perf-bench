# -*- coding: utf-8 -*-
"""Throwaway case for P0a: dense matmul ``C = A @ B`` (DISPOSABLE).

Exists only to validate the L0 core + L4 schema end-to-end — *not* a nitrix
op (P0a is decoupled from nitrix).  Matmul is chosen because it is universally
understood, output-independent (each ``C[i, j]`` depends only on row i / col j,
so fp64 subsampling would be valid), and it lets us exercise:

- multiple baselines under one case (``jnp-matmul``, ``jnp-einsum``,
  ``numpy-matmul``);
- the multi-framework sync hook (jax vs numpy) inside one run;
- the fp64 oracle + fidelity gate, including a deliberately low-precision
  baseline (``jnp-bf16``) that *fails* fidelity, to validate that a
  ``fidelity_failed`` row still carries its metrics and refuses only the ratio.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ._base import BuiltPoint, Case


def _build(param: Dict[str, Any]) -> BuiltPoint:
    m, k, n = param['m'], param['k'], param['n']
    rng = np.random.default_rng(param.get('seed', 0))
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    # On-device jax inputs (block so H2D is excluded from the timed region).
    ja = jax.block_until_ready(jnp.asarray(a))
    jb = jax.block_until_ready(jnp.asarray(b))

    # fp64 ground truth, computed once, outside any timed region.
    ref = a.astype(np.float64) @ b.astype(np.float64)

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (ja, jb) if framework == 'jax' else (a, b)

    baselines = {
        'jnp-matmul': ('jax', lambda x, y: jnp.matmul(x, y)),
        'jnp-einsum': ('jax', lambda x, y: jnp.einsum('mk,kn->mn', x, y)),
        'numpy-matmul': ('numpy', lambda x, y: np.matmul(x, y)),
        # Deliberately lossy: bf16 accumulation -> fails fidelity vs fp64.
        'jnp-bf16': (
            'jax',
            lambda x, y: jnp.matmul(
                x.astype(jnp.bfloat16), y.astype(jnp.bfloat16)
            ).astype(jnp.float32),
        ),
    }
    return BuiltPoint(
        baselines=baselines,
        inputs_for=inputs_for,
        fp64_reference=ref,
        ratio_reference='numpy-matmul',
    )


CASE = Case(
    name='dense_matmul',
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[
        {'m': 128, 'k': 128, 'n': 128, 'seed': 0},
        {'m': 512, 'k': 256, 'n': 512, 'seed': 0},
    ],
    representative={'m': 512, 'k': 256, 'n': 512, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
