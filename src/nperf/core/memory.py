# -*- coding: utf-8 -*-
"""Memory metrics (L1).

Peak HBM is read from the allocator of the framework that ran the op: jax /
numpy from ``jax.devices()[0].memory_stats()`` (only meaningful with
``XLA_PYTHON_CLIENT_PREALLOCATE=false``, recorded in provenance, DESIGN §1.1 —
else XLA's preallocation swamps the signal); a torch baseline from
``torch.cuda.max_memory_allocated`` (jax's counter cannot see torch's caching
allocator).  On CPU there is no HBM, so ``peak_hbm_mb`` returns ``None`` and
``host_rss_mb`` is the memory signal.

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


def peak_hbm_mb(framework: str = 'jax') -> Optional[float]:
    '''Peak device HBM in MB for the framework that *ran the op* (else None).

    A torch baseline's GPU memory lives in torch's caching allocator, which
    jax's ``peak_bytes_in_use`` never sees -- so a torch attempt must read
    ``torch.cuda.max_memory_allocated`` instead, or the row would claim a
    near-zero HBM that is simply jax's (idle) pool.  jax / numpy read jax's
    device stats as before.  Either way it is a process high-water mark, i.e.
    the attempt's peak under the subprocess runner (see the caveat above); on
    CPU there is no HBM and this is ``None`` (host RSS is the signal).'''
    if framework == 'torch':
        try:
            import torch

            if torch.cuda.is_available():
                return float(torch.cuda.max_memory_allocated()) / 1e6
        except Exception:
            return None
        return None
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
