# -*- coding: utf-8 -*-
"""Memory metrics (L1).

Peak HBM is read from ``jax.devices()[0].memory_stats()`` and is only
meaningful with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (recorded in
provenance, DESIGN §1.1) — otherwise XLA's preallocation swamps the signal.
On CPU there is no HBM, so ``peak_hbm_mb`` returns ``None`` and ``host_rss_mb``
is the memory signal.

NOTE (P0a limitation): ``peak_bytes_in_use`` is a *process* high-water mark;
JAX exposes no public per-attempt reset, so the value is the HWM up to the
measurement.  A clean per-attempt peak is a P1 item; the schema already carries
the field so the renderer is stable.
"""
from __future__ import annotations

import resource
from typing import Optional

import jax


def peak_hbm_mb() -> Optional[float]:
    '''Peak device ``bytes_in_use`` in MB, or ``None`` on CPU / no stats.'''
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    peak = stats.get('peak_bytes_in_use')
    return None if peak is None else float(peak) / 1e6


def host_rss_mb() -> float:
    '''Process resident-set high-water mark in MB (``ru_maxrss`` is KB on
    Linux).'''
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
