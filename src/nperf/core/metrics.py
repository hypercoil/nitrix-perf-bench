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
        # --- Registration recovery quality (REGISTRATION_RECOVERY) ---------
        # Scored per-baseline from a *planted-warp* point with a known ground
        # truth (BuiltPoint.recovery), NOT an oracle two-array compare.  How
        # WELL each tool recovered the known transform -- reported beside speed
        # so the report shows the speed/accuracy tradeoff, not speed alone.
        # ``recovery_ncc`` is the UNIFORM column (every tool: warped-vs-fixed
        # alignment); TRE / warp / jac are the rigorous transform-based scores
        # (where the recovered field is extractable -- nitrix/ANTs/dipy).
        Metric('recovery_ncc', 'ncc', 'higher'),   # warped vs fixed (uniform)
        Metric('recovery_tre', 'mm', 'lower'),     # TRE @ landmarks
        Metric('recovery_warp', 'mm', 'lower'),    # RMS field-vs-GT over mask
        Metric('recovery_jacmin', 'detJ', 'higher'),  # min detJ (<0 = fold)
        # 'energy' deferred (distinct protocol, DESIGN §L1).
    )
}

