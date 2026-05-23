# -*- coding: utf-8 -*-
"""First real nitrix case (P0b): dense ``semiring_matmul``.

This is the case that supersedes the hand-built
``bench/PERF_SEMIRING_MATMUL.md`` in nitrix.  It asks the question that report
exists to answer: **is the Pallas / Triton kernel worth its complexity over the
pure-JAX reference**, on the algebras ``jnp.matmul`` cannot accelerate (``log``
/ ``tropical`` / ``euclidean`` run on CUDA cores, not tensor cores)?

What it exercises beyond the throwaway:

- a **real nitrix import** (the system under test enters as a path source);
- three same-framework baselines that are the *same op, different strategy*:
  ``nitrix-jax`` (the streaming ``fori_loop`` reference), ``nitrix-pallas``
  (the Triton kernel), and ``naive-dense`` (materialise-then-reduce).  The
  ratio that matters is *vs the reference*, so
  ``ratio_reference = 'nitrix-jax'``;
- a per-algebra ceiling (``jnp-matmul``) added **only for ``real``**, where the
  honest fast path is tensor cores (cf. the in-tree report's note);
- a **scalable fp64 oracle**: the case's own reference op
  (``reference_semiring_matmul``) evaluated in true fp64.  It is a
  ``fori_loop`` over the contraction axis, so the oracle costs O(m·n) memory —
  a naive dense ``(m, k, n)`` broadcast would be O(m·k·n) and blow up at
  k = 1024.

The ``naive-dense`` baseline is the *space–time counterpoint* nitrix never
shipped (its in-tree bench had a single baseline): it expresses
``C = reduce_k(combine(A[:, :, None], B[None, :, :]))`` — conceptually the full
``(m, k, n)`` combine tensor reduced over the contraction axis in one
vectorised pass.  Measured on an A10G it is indeed **much faster** at steady
state (≈9× the JAX reference at 512³ log) — but at two real costs the report
surfaces: **elevated ``peak_hbm``** (the materialised operand needs ~68–85 MB
vs the streaming kernels' 2.6–23 MB — 3–26× depending on size/algebra; XLA
*tiles* the reduction so it is not the full O(m·k·n) blow-up, but is markedly
larger than the streaming O(m·n)) and a **pathological cold ``compile_time``**
(XLA fusing a large reduction over the expanded operand — ~300 s cold per
reduction point, ~580 s for 512³ logsumexp).  Both are *findings*, not bugs;
``compile_time`` and ``peak_hbm`` being first-class metrics is why they are
visible.  These costs are only honest under the subprocess runner (``peak_hbm``
is per-attempt only with process isolation; cold compile needs a fresh process
— see ``core/memory.py`` / SCHEMA_AND_LIFECYCLE §B).

Oracle honesty note: the fp64 oracle and the ``nitrix-jax`` baseline share the
reference code path (the baseline *is* the reference, in fp32).  That is by
design — scoring fp32-reference against fp64-reference measures pure round-off,
and scoring the *Pallas kernel* / *naive-dense* against the fp64 reference is
exactly the "does this strategy match the reference math" check (the same
contract nitrix's own backend-parity tests enforce).  Correctness of the
reference itself is nitrix's suite, not this repo's (DESIGN §1, "performance
only").

The outputs are zero-centred for ``real`` and shifted-but-not-relative for
``log`` / ``tropical`` / ``euclidean``; either way the fidelity headline is
``rel_to_tol`` (tolerance-relative), never a bare ``max_rel``
(SCHEMA_AND_LIFECYCLE §C).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
from nitrix.semiring import (
    EUCLIDEAN,
    LOG,
    REAL,
    TROPICAL_MAX_PLUS,
    reference_semiring_matmul,
    semiring_matmul,
)

from ._base import BuiltPoint, Case

# Algebras under test, keyed by their nitrix ``Semiring.name`` so a param point
# stays a plain filterable string.  ``boolean`` / ``tropical_min_plus`` exist
# in nitrix but add no new perf shape here; the four below cover the distinct
# kernel behaviours (product+sum, logsumexp, max+sum, squared-diff+sqrt).
_ALGEBRAS = {
    s.name: s for s in (REAL, LOG, TROPICAL_MAX_PLUS, EUCLIDEAN)
}

# Naive-dense axis reduction + finalize, per algebra.  The *combine* is reused
# straight from the algebra (``semiring.binary_op.combine`` broadcasts), so
# only the contraction-axis reduction and the finalize need a mapping — the
# monoid exposes an *online* ``update``, not an axis reduce.  These mirror the
# monoids in ``nitrix.semiring.algebras`` (sum / logsumexp / max / sum-sqrt).
_NAIVE_REDUCE: Dict[str, Tuple[Callable[..., Any], Callable[[Any], Any]]] = {
    'real': (jnp.sum, lambda x: x),
    'log': (jsp.logsumexp, lambda x: x),
    'tropical_max_plus': (jnp.max, lambda x: x),
    'euclidean': (
        jnp.sum, lambda x: jnp.sqrt(jnp.maximum(x, jnp.zeros_like(x)))
    ),
}


def _naive_dense_fn(
    semiring: Any, reduce_k: Callable[..., Any], finalize: Callable[[Any], Any]
) -> Callable[[Any, Any], Any]:
    '''Build the materialise-then-reduce baseline for one algebra.

    ``M[i, k, j] = combine(A[i, k], B[k, j])`` is the full ``(m, k, n)`` tensor
    (the O(m·k·n) memory cost); the reduction over ``k`` and the finalize then
    produce ``C[i, j]``.
    '''

    def run(A: Any, B: Any) -> Any:
        combined = semiring.binary_op.combine(A[:, :, None], B[None, :, :])
        return finalize(reduce_k(combined, axis=1))

    return run


def _build(param: Dict[str, Any]) -> BuiltPoint:
    m, k, n = param['m'], param['k'], param['n']
    name = param['algebra']
    alg = _ALGEBRAS[name]
    rng = np.random.default_rng(param.get('seed', 0))
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((k, n)).astype(np.float32)

    # On-device fp32 inputs; block so H2D is excluded from the timed region.
    ja = jax.block_until_ready(jnp.asarray(a))
    jb = jax.block_until_ready(jnp.asarray(b))

    # fp64 ground truth: the case's reference op, in true fp64, on the *same*
    # input values the baselines see.  Requires jax_enable_x64 (set in the
    # driver and recorded in provenance); enabled defensively here so the case
    # is also correct when built directly from a test.
    jax.config.update('jax_enable_x64', True)
    ref = np.asarray(
        jax.block_until_ready(
            reference_semiring_matmul(
                jnp.asarray(a, dtype=jnp.float64),
                jnp.asarray(b, dtype=jnp.float64),
                semiring=alg,
            )
        ),
        dtype=np.float64,
    )

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        return (ja, jb)  # all baselines here are jax

    reduce_k, finalize = _NAIVE_REDUCE[name]
    baselines = {
        'nitrix-jax': (
            'jax',
            lambda x, y: semiring_matmul(x, y, semiring=alg, backend='jax'),
        ),
        'nitrix-pallas': (
            'jax',
            lambda x, y: semiring_matmul(
                x, y, semiring=alg, backend='pallas-cuda'
            ),
        ),
        'naive-dense': ('jax', _naive_dense_fn(alg, reduce_k, finalize)),
    }
    if alg is REAL:
        # The honest fast path for the real semiring is tensor cores; include
        # it as a ceiling.  (Precision is forced to 'highest' in the driver so
        # this is true fp32, not a silent TF32 downgrade -- see provenance.)
        baselines['jnp-matmul'] = ('jax', lambda x, y: jnp.matmul(x, y))

    return BuiltPoint(
        baselines=baselines,
        inputs_for=inputs_for,
        fp64_reference=ref,
        ratio_reference='nitrix-jax',
    )


# Coverage-tier shape ladder.  Stops at 512³ on purpose: the ``naive-dense``
# baseline's *cold compile* grows pathologically with the contraction size (XLA
# fusing a large reduction over the materialised combine tensor — ~10 min for a
# single 512³ logsumexp point on an A10G) and its memory grows O(m·k·n), so a
# routine multi-baseline sweep past 512³ is impractical.  Larger shapes are a
# decision-tier sweep, run on demand — not the default coverage ladder.
_SHAPES = [(256, 256, 256), (512, 512, 512)]

CASE = Case(
    name='semiring_matmul',
    # Each C[i, j] depends only on row i of A and column j of B -> a bounded,
    # identifiable input subset, so fp64 subsampling would be *valid* (annex
    # §C).  We use fp64_full anyway because the fori_loop oracle is cheap.
    output_independent=True,
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[
        {'m': m, 'k': k, 'n': n, 'algebra': name, 'seed': 0}
        for (m, k, n) in _SHAPES
        for name in _ALGEBRAS
    ],
    # Decision-relevant anchor: a non-real algebra (jnp can't do it) at a
    # mid size -- this is the cell where "is Pallas worth it" actually bites.
    representative={'m': 512, 'k': 512, 'n': 512, 'algebra': 'log', 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
