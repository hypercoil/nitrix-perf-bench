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

import argparse
import json
from pathlib import Path

import jax

from .core import capture, write_jsonl
from .measure import CASES, measure_attempt


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
