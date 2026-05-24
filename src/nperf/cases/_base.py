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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


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
    # the public nitrix op this case measures (its op_matrix qualname); the
    # single home for the case -> op mapping the op_matrix feed and the
    # decision-input bundle both read.  None for the throwaway smoke case.
    op_qualname: Optional[str] = None
