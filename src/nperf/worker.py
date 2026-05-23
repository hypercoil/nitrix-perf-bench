# -*- coding: utf-8 -*-
"""Single-attempt measurement worker (`python -m nperf.worker --spec FILE`).

**One OS process per attempt.**  This is the load-bearing P1 change: because
the process is fresh, the device/host memory high-water marks
(`peak_bytes_in_use`, `ru_maxrss`) measured here *are* this attempt's peak —
not a value contaminated by whatever earlier attempts allocated in a shared
process (the in-process driver's caveat; see `core/memory.py` /
SCHEMA_AND_LIFECYCLE §B).  It also means a crash (segfault, OOM-kill) takes
down only this attempt: the orchestrator sees the dead process and records a
failure row, and the sweep continues.

Contract: read a spec JSON, set the numeric policy, build the case point, run
**one** baseline through the shared `measure_attempt`, and write exactly one L4
row to `spec["result_path"]`.  Provenance is captured *here* (this process is
the one on the device), so each worker stamps its own device — that is what
makes a multi-platform run's rows self-describing.
"""
from __future__ import annotations

import os

# CPU affinity must be applied **before jax/XLA import** so XLA sizes its CPU
# thread pool to this slot's cores -- that is what keeps parallel-CPU timing
# honest (disjoint cores, no contention; the scheduler assigns the group, see
# schedule.py / annex §E).  Hence the imports below this block are deliberately
# not at module top (ruff E402 is silenced for exactly those lines).
_CORES = os.environ.get('NPERF_CPU_CORES')
if _CORES:
    _core_set = {int(c) for c in _CORES.split(',') if c != ''}
    if _core_set and hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, _core_set)
        except OSError:
            pass
        os.environ.setdefault('OMP_NUM_THREADS', str(len(_core_set)))

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402

from .core import capture, write_jsonl  # noqa: E402
from .measure import CASES, measure_attempt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--spec', required=True,
                    help='JSON spec: run_id, case, param_point, baseline, '
                         'platform, warmup, repeats, result_path')
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())

    # Same fair, recorded numeric policy as the driver (annex §C): true fp32
    # matmul, and x64 so the fp64 oracle built below is genuinely double.
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_enable_x64', True)

    prov = capture()
    prov['measurement_isolation'] = 'subprocess'  # honest per-attempt memory
    if _CORES and hasattr(os, 'sched_getaffinity'):
        prov['cpu_affinity'] = sorted(os.sched_getaffinity(0))  # pinned slot

    case = CASES[spec['case']]
    param = spec['param_point']
    built = case.build(param)
    rec = measure_attempt(
        case, param, built, spec['baseline'],
        platform=spec['platform'], run_id=spec['run_id'], prov=prov,
        warmup=spec['warmup'], repeats=spec['repeats'],
    )
    write_jsonl([rec], spec['result_path'])


if __name__ == '__main__':
    main()
