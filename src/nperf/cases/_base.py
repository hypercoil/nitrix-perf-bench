# -*- coding: utf-8 -*-
"""Case + BuiltPoint protocol (L2).

A ``Case`` declares an op under test: the param points it spans, its
*representative* point (the coverage-tier anchor, DESIGN §1/§4), whether its
outputs are *element-wise independent* (gates fp64 subsampling, annex §C), the
metrics it supports, and a ``build(param)`` that materialises one
``BuiltPoint``.

A ``BuiltPoint`` carries the competing implementations and the shared oracle
for one param point: per-framework on-device inputs, the baseline registry
(``name -> (framework, run_fn)``), the fp64 ground-truth output (computed once,
shared across baselines), and which baseline ratios are taken against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class BuiltPoint:
    # name -> (framework, run_fn(*args) -> output)
    baselines: Dict[str, Tuple[str, Callable[..., Any]]]
    # framework -> on-device args (host->device transfer excluded from timing)
    inputs_for: Callable[[str], Tuple[Any, ...]]
    # ground-truth output, host fp64, computed once for this param point
    fp64_reference: Any
    # baseline name that ratios are computed against
    ratio_reference: str


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
