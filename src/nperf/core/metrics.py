# -*- coding: utf-8 -*-
"""Metric registry (L1).

A metric is a name + unit + direction (does lower or higher win?) + kind.  The
direction is what lets the ratio / regression layer know which way is a "win"
without hard-coding it per renderer; the unit is the single source of truth the
driver stamps onto each value.  Metrics are descriptors; the values are
collected by the driver and stored under ``AttemptRecord.metrics`` (the
``fidelity`` metric is the exception — it lives in the ``fidelity`` block (it
gates the ratio); it is catalogued here so its unit / direction / gate
``threshold`` have one home).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    direction: str  # 'lower' | 'higher' (which way is better)
    kind: str = 'scalar'  # 'scalar' | 'distribution' | 'fidelity'
    threshold: Optional[float] = None  # pass/fail boundary (fidelity)


METRICS: Dict[str, Metric] = {
    m.name: m
    for m in (
        Metric('steady_time', 's', 'lower', kind='distribution'),
        Metric('compile_time', 's', 'lower'),
        Metric('peak_hbm', 'MB', 'lower'),
        Metric('host_rss', 'MB', 'lower'),
        Metric('throughput', 'elem/s', 'higher'),
        # Stored in the `fidelity` block, not `metrics`; catalogued for its
        # unit/direction/gate-threshold (pass <=> rel_to_tol <= 1; annex §C).
        Metric('fidelity', '×tol', 'lower', kind='fidelity', threshold=1.0),
        # 'energy' deferred (distinct protocol, DESIGN §L1).
    )
}

