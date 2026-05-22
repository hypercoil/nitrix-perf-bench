# -*- coding: utf-8 -*-
"""Provenance capture (DESIGN §1.1).

Every L4 row carries a provenance block so a measurement is attributable and
reproducible.  Fields nitrix-perf-bench cannot yet introspect cheaply (CPU
governor, GPU clock-lock, NUMA binding, resolved dep versions) are recorded as
``None`` rather than omitted — "record it even if unknown" so the gap is
visible, not silent.  Filling them is a P1 item.
"""
from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

import jax


def _git(path: str) -> Dict[str, Optional[Any]]:
    try:
        sha = subprocess.run(
            ['git', '-C', path, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', '-C', path, 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
        return {'sha': sha or None, 'dirty': dirty}
    except Exception:
        return {'sha': None, 'dirty': None}


def capture(*, nitrix_path: str = '../nitrix') -> Dict[str, Any]:
    '''Snapshot the host + toolchain + source state for a run.'''
    try:
        dev = jax.devices()[0]
        device = {'kind': dev.device_kind, 'platform': dev.platform}
    except Exception:
        device = {'kind': None, 'platform': None}
    return {
        'timestamp': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'bench': _git('.'),
        'nitrix': _git(nitrix_path),
        'jax_version': jax.__version__,
        'jax_backend': device['platform'],
        'device': device,
        'python': sys.version.split()[0],
        'os': platform.platform(),
        'precision_policy': str(
            getattr(jax.config, 'jax_default_matmul_precision', None)
        ),
        'xla_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        'xla_flags': os.environ.get('XLA_FLAGS'),
        'compile_cache': 'disabled',  # P0a policy (DESIGN §L0 / annex §D)
        # §1.1 fields not yet introspected -- recorded as null (P1).
        'cpu_model': None,
        'cpu_governor': None,
        'gpu_clock_locked': None,
        'numa_binding': None,
        'resolved_versions': None,
    }


def make_run_id(prov: Dict[str, Any]) -> str:
    '''A run id: timestamp + short bench SHA (groups one invocation).'''
    ts = prov.get('timestamp', '').replace(':', '').replace('-', '')
    sha = (prov.get('bench', {}) or {}).get('sha') or 'nosha'
    return f'{ts}__{sha[:7]}'
