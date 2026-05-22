# -*- coding: utf-8 -*-
"""L0 measurement core + L1 metric/fidelity/results primitives.

Framework-agnostic: this layer knows nothing about nitrix, cases, or
rendering (DESIGN §2).
"""
from .fidelity import compare
from .memory import host_rss_mb, peak_hbm_mb
from .metrics import METRICS, Metric
from .provenance import capture, make_run_id
from .results import (
    SCHEMA_VERSION,
    AttemptRecord,
    Status,
    read_jsonl,
    write_jsonl,
)
from .stats import Distribution
from .sync import SYNC, jax_sync, numpy_sync
from .timer import bench_call

__all__ = [
    'compare',
    'host_rss_mb',
    'peak_hbm_mb',
    'METRICS',
    'Metric',
    'capture',
    'make_run_id',
    'SCHEMA_VERSION',
    'AttemptRecord',
    'Status',
    'read_jsonl',
    'write_jsonl',
    'Distribution',
    'SYNC',
    'jax_sync',
    'numpy_sync',
    'bench_call',
]
