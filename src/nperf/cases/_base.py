# -*- coding: utf-8 -*-
"""Case + BuiltPoint protocol (L2).

A ``Case`` declares an op under test: the param points it spans, its
*representative* point (the coverage-tier anchor, DESIGN §1/§4), whether its
outputs are *element-wise independent* (gates fp64 subsampling, annex §C), the
metrics it supports, and a ``build(param)`` that materialises one
``BuiltPoint``.

A ``BuiltPoint`` carries the competing implementations and the shared oracle
for one param point: per-framework on-device inputs, the baselines
(``case-local name -> (provider_id, run_fn)``; the ``provider_id`` keys the
cross-case provider registry in ``nperf.providers`` for framework + env
isolation), the fp64 ground-truth output (computed once, shared across
baselines), and which baseline ratios are taken against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def to_cupy(*arrays: Any) -> Tuple[Any, ...]:
    '''Host arrays -> on-device CuPy arrays, synchronised so the H2D transfer
    sits *outside* the timed region (mirrors the ``jax.block_until_ready`` on
    the jax inputs in a case ``build``).  ``cupy`` is imported lazily, so only
    the cupy worker (the refs-cupy env) ever needs it -- a jax/numpy worker
    that calls a case's ``inputs_for`` for its own framework never imports it.
    Used by the audit cases' GPU reference (Phase B).'''
    import cupy as cp

    out = tuple(cp.asarray(a) for a in arrays)
    cp.cuda.runtime.deviceSynchronize()
    return out


@dataclass(frozen=True)
class SlowBaseline:
    '''A baseline whose *measurement* is pathologically expensive (not its
    steady state) -- e.g. a cold compile of minutes -- so a dev-cycle run may
    skip it via ``--skip-slow`` (`run.py`).

    ``reason`` is the audit trail (why + the measured cost, like a provider's
    ``pixi_reason``): slowness is *evidence-based and hardware-dependent*, so
    record the number and the device it was seen on, and re-evaluate on
    re-bench rather than treating the entry as eternal.

    Known limitation (deferred, no action yet): slowness is currently
    **platform-flat** -- ``--skip-slow`` drops the baseline on *every*
    platform, even when it is slow on only one.  Measured counter-example:
    ``naive-dense`` costs ~432s to compile on the L4 GPU but only ~0.3s on CPU,
    so a GPU-driven skip needlessly drops the cheap, useful CPU measurement.  A
    ``platforms: Optional[Tuple[str, ...]]`` field to scope the skip is the
    refinement; deferred because dev cycles run both platforms together and the
    GPU compile is the long pole, so the flat skip is acceptable for now.'''
    baseline: str
    reason: str


@dataclass(frozen=True)
class ApproxBaseline:
    '''A baseline that computes the case's task only **approximately** -- a
    fidelity/speed tradeoff, not a bug.  Its fidelity is *reported, not gated*:
    the row stays ``OK`` and still earns a ratio, but its fidelity block is
    marked ``'approximate'`` carrying the measured ``rel_to_tol`` -- the
    approximation magnitude *is* the signal, read against the speed (the
    quantised-inference case: practitioners knowingly trade accuracy for
    throughput).  This is distinct from a true reference, which must pass the
    case's tight gate, and from a fidelity *failure*, which refuses the ratio.

    ``reason`` is the audit trail -- what the approximation is, its measured
    magnitude, and the device it was seen on -- like ``SlowBaseline``:
    evidence-based, re-checked on re-bench rather than treated as eternal.
    '''
    baseline: str
    reason: str


@dataclass
class BuiltPoint:
    # case-local baseline name -> (provider_id, run_fn(*args) -> output);
    # provider_id keys nperf.providers (framework + env isolation).
    baselines: Dict[str, Tuple[str, Callable[..., Any]]]
    # framework -> on-device args (host->device transfer excluded from timing)
    inputs_for: Callable[[str], Tuple[Any, ...]]
    # ground-truth output, host fp64, computed once for this param point.
    # ``None`` means there is **no cross-implementation oracle** (the baselines
    # compute the same task but not bit-identical results -- e.g. a different
    # boundary convention); the attempt is then OK with an *inconclusive*
    # fidelity block instead of a gate comparison (see measure_attempt).
    fp64_reference: Any
    # baseline name that ratios are computed against
    ratio_reference: str
    # when fp64_reference is None: the human reason the comparison is N/A
    # (recorded as fidelity.reason): why the baselines legitimately differ.
    fidelity_note: Optional[str] = None


@dataclass
class Case:
    name: str
    output_independent: bool
    metrics: List[str]
    param_points: List[Dict[str, Any]]
    representative: Dict[str, Any]
    build: Callable[[Dict[str, Any]], BuiltPoint]
    # per-case fidelity tolerance (np.allclose semantics)
    rtol: float = 1e-3
    atol: float = 1e-4
    # baselines whose *measurement* is pathologically slow (cold compile, …);
    # ``--skip-slow`` drops these for fast dev cycles (recorded as skipped
    # rows, and the run is stamped ``coverage_mode = fast`` so the op_matrix
    # feed / gate refuse to treat it as authoritative -- run a full sweep at
    # sprint end).
    slow_baselines: Tuple[SlowBaseline, ...] = field(default_factory=tuple)
    # baselines that compute the task only *approximately* (a fidelity/speed
    # tradeoff, e.g. a 4SED distance transform or a quantised kernel): their
    # fidelity is reported, not gated -- the row stays OK with an
    # ``'approximate'`` fidelity block and still earns a ratio, so the
    # accuracy-vs-speed tradeoff is visible, not dropped (ApproxBaseline).
    approximate_baselines: Tuple[ApproxBaseline, ...] = field(
        default_factory=tuple)
    # the public nitrix op this case measures (its op_matrix qualname); the
    # single home for the case -> op mapping the op_matrix feed and the
    # decision-input bundle both read.  None for the throwaway smoke case.
    op_qualname: Optional[str] = None
