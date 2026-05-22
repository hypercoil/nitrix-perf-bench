# -*- coding: utf-8 -*-
"""Metric registry (L1).

A metric is a name + unit + direction (does lower or higher win?).  The
direction is what lets the ratio / regression layer know which way is a
"win" without hard-coding it per renderer.  Metrics are descriptors; the
values are collected by the driver and stored under ``AttemptRecord.metrics``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    direction: str  # 'lower' | 'higher'


METRICS: Dict[str, Metric] = {
    m.name: m
    for m in (
        Metric('steady_time', 's', 'lower'),
        Metric('compile_time', 's', 'lower'),
        Metric('peak_hbm', 'MB', 'lower'),
        Metric('host_rss', 'MB', 'lower'),
        Metric('throughput', 'elem/s', 'higher'),
        # 'energy' deferred (distinct protocol, DESIGN §L1).
    )
}
