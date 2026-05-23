# -*- coding: utf-8 -*-
"""Memory metrics (L1).

Peak HBM is read from ``jax.devices()[0].memory_stats()`` and is only
meaningful with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (recorded in
provenance, DESIGN §1.1) — otherwise XLA's preallocation swamps the signal.
On CPU there is no HBM, so ``peak_hbm_mb`` returns ``None`` and ``host_rss_mb``
is the memory signal.

⚠️  LOUD CAVEAT — both memory signals here are *process-wide high-water marks*
that **never reset** within a run, so they are only trustworthy per-attempt in
a **single-attempt process**:

- ``peak_bytes_in_use`` (HBM) is a monotonic process maximum; JAX exposes no
  public per-attempt reset.
- ``ru_maxrss`` (host RSS) is likewise the process max RSS, not the attempt's.

In the current **in-process** driver, once any attempt allocates a large buffer
every later row inherits that floor — only the *jumps* attribute to the attempt
that caused them.  The real fix is **one OS process per attempt** (the P1
subprocess workers, SCHEMA_AND_LIFECYCLE §B): then each process's HWM *is* the
attempt's peak.  The schema already carries the field, so the renderer is
stable across the fix; only the isolation changes.  Until then, treat per-row
memory as a process HWM, not an isolated footprint.
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
