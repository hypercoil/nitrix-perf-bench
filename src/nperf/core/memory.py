# -*- coding: utf-8 -*-
"""Memory metrics (L1).

Peak HBM is read from ``jax.devices()[0].memory_stats()`` and is only
meaningful with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (recorded in
provenance, DESIGN §1.1) — otherwise XLA's preallocation swamps the signal.
On CPU there is no HBM, so ``peak_hbm_mb`` returns ``None`` and ``host_rss_mb``
is the memory signal.

⚠️  CAVEAT — both memory signals here are *process-wide high-water marks* that
**never reset** within a process, so they are only trustworthy per-attempt when
each attempt gets its **own process**:

- ``peak_bytes_in_use`` (HBM) is a monotonic process maximum; JAX exposes no
  public per-attempt reset.
- ``ru_maxrss`` (host RSS) is likewise the process max RSS, not the attempt's.

This is **resolved by default**: the P1 runner (`run.py` / `worker.py`) gives
each attempt its own subprocess, so the worker's HWM *is* that attempt's peak
(`provenance.measurement_isolation == 'subprocess'`).  The **only** case where
the high-water-mark caveat still bites is the ``--in-process`` driver (kept for
fast CPU smoke), where one process runs every attempt and the value only ever
rises — there, read the *jumps* (they attribute to the attempt that caused
them), not the absolute per-row value.  The renderer states whichever applies
on every report (annex §B).
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
