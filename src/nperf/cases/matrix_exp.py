# -*- coding: utf-8 -*-
"""Registration primitive: ``nitrix.linalg.matrix_exp`` vs scipy / jax expm.

The matrix exponential ``e^A`` -- the bottleneck of ``affine_exp`` (the affine
Lie-group chart) and a general ``nitrix.linalg`` primitive.  nitrix computes it
by scaling-and-squaring with a Taylor series: ``e^A = (T_k(A / 2^s))^(2^s)``
for ``s = n_squarings`` doublings and a degree-``k = taylor_order`` Taylor
polynomial -- a stack of ~``k + s`` matmuls, **no matrix factorisation**.  The
comparison worth measuring (B18) is that against the **Padé** path:

- ``scipy.linalg.expm`` / ``jax.scipy.linalg.expm`` use Padé, which needs a
  dense **linear solve** (LU) -- ``jax``'s ``expm`` was confirmed to route
  through ``solve``.  So the algorithmic difference is matmul-stack (nitrix)
  vs matmuls-plus-a-solve (Padé); both are ``O(n^3)``.  On this box jax's Padé
  ``expm`` ran on the GPU fine, so the rows are a like-for-like GPU comparison.
  (We do *not* know enough about the cause to claim the matmul-only path is
  uniquely GPU-native -- that's a this-machine observation, not a portable
  property.)
- Measured (this L4): nitrix is **1.4-3.4x faster than ``jax.scipy.expm``** --
  the margin (the saved Padé solve) is widest at small ``n`` (~3x) and narrows
  to ~1.4x at ``n=1024`` as the shared O(n^3) matmul cost dominates.  vs
  ``scipy`` (CPU) nitrix wins 21-73x at ``n>=256`` but *loses* at ``n=16``
  (GPU launch overhead dwarfs a 16x16 expm).

**Warranted comparison + pinned accuracy.**  Inputs are random matrices scaled
to a fixed spectral norm (``|A|_2 = 4`` -- a realistic affine-generator scale,
and within the scaling-and-squaring's accurate regime), so nitrix matches
``scipy.linalg.expm`` (the fp64 oracle + CPU floor) to **fp32 round-off**: the
measured worst absolute error is ~1e-4 at ``n<=256`` rising to ~4e-4 at
``n=1024`` (honest fp32 accumulation over the ~20-matmul chain), so the gate is
``atol=5e-4`` -- tight enough to catch an algorithmic regression (a wrong
assembly / far-too-few squarings blows well past it), loose enough not to flag
fp32.  nitrix runs fp32 (what affine registration runs on the GPU); the oracle
is scipy in fp64.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla
import numpy as np
import scipy.linalg as sla
from nitrix.linalg import matrix_exp

from ._base import BuiltPoint, Case

_NORM = 4.0  # target spectral norm (realistic affine-generator scale)


def _generator(n: int, seed: int, norm: float = _NORM) -> np.ndarray:
    '''A random ``n x n`` matrix scaled to spectral norm ``norm`` (fp32) -- the
    affine-generator regime where scaling-and-squaring is accurate.'''
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)).astype(np.float32)
    return (a / np.linalg.norm(a, 2) * norm).astype(np.float32)


def _scipy_expm(a: Any) -> np.ndarray:
    '''Padé matrix exponential (CPU floor + fp64 oracle).'''
    return sla.expm(np.asarray(a))


def _jax_expm(a: Any) -> Any:
    '''jax's Padé expm (the GPU twin) -- routes through a dense ``solve`` (LU),
    which ran on the GPU on this box.  The measured gap to nitrix is the cost
    of that solve atop the matmuls.'''
    return jsla.expm(a)


def _build(param: Dict[str, Any]) -> BuiltPoint:
    n = int(param['n'])
    a = _generator(n, param.get('seed', 0), param.get('norm', _NORM))
    aj = jax.block_until_ready(jnp.asarray(a))

    ref = _scipy_expm(a.astype(np.float64))  # fp64 oracle

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (a,) if framework == 'numpy' else (aj,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: matrix_exp(x)),
        'scipy.linalg.expm': ('scipy', _scipy_expm),  # CPU floor + oracle
        'jax.scipy.linalg.expm': ('jax', _jax_expm),  # GPU twin (Padé+solve)
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# Single matrices across n (the matmul-stack scaling); norm pinned so accuracy
# is fp32-tight at every n.  The affine regime is small (n=3/4); these moderate
# n make the matmul work -- and the GPU-native-vs-Padé-solve gap -- measurable.
_NS = [16, 64, 256]

# Scale tier (COVERAGE_MANDATE §2.6): larger n where the O(n^3) matmul stack
# dominates and the Padé path's dense solve is most expensive / most likely to
# fall off the GPU.  Norm pinned, so the only axis is n.
_LARGE = [512, 1024]

CASE = Case(
    name='matrix_exp',
    op_qualname='nitrix.linalg.matrix_exp',
    output_independent=False,  # each entry of e^A depends on the whole matrix
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'n': n, 'seed': 0} for n in _NS],
    large_param_points=tuple({'n': n, 'seed': 0} for n in _LARGE),
    representative={'n': 64, 'seed': 0},
    complexity=(
        'both O(n^3): nitrix is a ~(taylor_order + n_squarings) ~= 20 matmul '
        'stack (no factorisation); scipy/jax expm is Padé + a dense LU solve. '
        'Measured (this L4): nitrix 1.4-3.4x faster than jax expm on GPU (the '
        'saved solve; margin narrows as the shared O(n^3) matmul dominates at '
        'n=1024), 21-73x vs scipy CPU at n>=256 but slower at n=16 (launch '
        'overhead). HBM O(n^2) (a few n x n temporaries), flat ~90-105 MB.'
    ),
    build=_build,
    rtol=1e-3,
    atol=5e-4,  # honest fp32 matrix-exp round-off (~1-4e-4); see module doc
)
