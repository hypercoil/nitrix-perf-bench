# -*- coding: utf-8 -*-
"""Runner / orchestrator (L3).

Default mode (**subprocess**): one OS process per attempt, spawned via a
pluggable interpreter (`worker.py`).  This is the P1 runner — it makes
per-attempt memory honest (each worker's high-water mark *is* its attempt's
peak; see `core/memory.py`) and isolates crashes (a dead worker becomes a
failure row, the sweep continues).  Workers run **serially** for now; the
device lock for parallel-per-GPU scheduling (annex §E) lands with the
parallelisation slice.

`--in-process` keeps the P0 driver (no spawn cost, faster for CPU smoke) — but
its memory metrics are process high-water marks, so the renderer flags them.

`--render-from` re-renders a report from saved L4 rows (no measurement).

Worker interpreter resolution (per `--platform`): `NPERF_PYTHON_<PLATFORM>`,
then `NPERF_WORKER_PYTHON`, else this interpreter (the uv default when you run
the orchestrator under the matching env; for a platform this env can't provide,
e.g. GPU from a CPU orchestrator, point `NPERF_PYTHON_JAX_CUDA12` at that env's
python, or run the orchestrator itself under `uv run --group <platform>`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from .core import (
    AttemptRecord,
    Status,
    capture,
    make_run_id,
    read_jsonl,
    write_jsonl,
)
from .measure import (
    CASES,
    attach_ratios,
    classify_message,
    measure_attempt,
    platform_from,
)
from .report import render_markdown

_SRC = str(Path(__file__).resolve().parents[1])  # `src` dir, for PYTHONPATH


# --------------------------------------------------------------------------- #
# Subprocess orchestration                                                    #
# --------------------------------------------------------------------------- #
def _worker_argv(platform: str) -> List[str]:
    override = (
        os.environ.get('NPERF_PYTHON_' + platform.upper().replace('-', '_'))
        or os.environ.get('NPERF_WORKER_PYTHON')
    )
    python = override or sys.executable
    return [python, '-m', 'nperf.worker']


def _worker_env(platform: str) -> Dict[str, str]:
    env = dict(os.environ)
    # nperf must be importable even when the interpreter is a prebuilt env that
    # never pip-installed it (nitrix is already importable in those by design).
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = _SRC + (os.pathsep + existing if existing else '')
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # honest memory_stats
    # Pin the worker's backend explicitly per target platform.
    env['JAX_PLATFORMS'] = 'cuda' if platform == 'jax-cuda12' else 'cpu'
    return env


def _synthesize_failure(
    base: Dict[str, Any], proc: subprocess.CompletedProcess[str]
) -> AttemptRecord:
    '''Worker exited without a row -> classify the death into a row.'''
    tail = (proc.stderr or '')[-2000:]
    rc = proc.returncode
    if rc < 0:  # killed by a signal; SIGKILL (9) is ~always the OOM-killer
        sig = -rc
        status = Status.OOM if sig == 9 else Status.COMPILE_ERROR
        return AttemptRecord(
            **base, status=status,
            failure_detail={
                'reason': 'worker_killed', 'signal': sig, 'stderr_tail': tail,
            },
        )
    status, detail = classify_message(tail or f'worker exited {rc}')
    detail['returncode'] = rc
    detail['stderr_tail'] = tail
    return AttemptRecord(**base, status=status, failure_detail=detail)


def _spawn_worker(
    spec: Dict[str, Any], platform: str, *, framework: str, timeout: float
) -> AttemptRecord:
    base = dict(
        run_id=spec['run_id'], case=spec['case'],
        param_point=spec['param_point'], baseline=spec['baseline'],
        platform=platform, framework=framework, provenance={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / 'row.jsonl'
        spec_path = Path(tmp) / 'spec.json'
        spec_path.write_text(
            json.dumps({**spec, 'result_path': str(result_path)})
        )
        argv = _worker_argv(platform) + ['--spec', str(spec_path)]
        try:
            proc = subprocess.run(
                argv, env=_worker_env(platform), capture_output=True,
                text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return AttemptRecord(
                **base, status=Status.TIMEOUT,
                failure_detail={'limit_s': timeout},
            )
        if result_path.exists():
            rows = read_jsonl(result_path)
            if rows:
                return AttemptRecord.from_json(rows[0])
        return _synthesize_failure(base, proc)


def run_case_subprocess(
    case: Any, *, platform: str, warmup: int, repeats: int, run_id: str,
    timeout: float,
) -> List[AttemptRecord]:
    records: List[AttemptRecord] = []
    for param in case.param_points:
        # Build once on the orchestrator to enumerate the point's baselines +
        # frameworks + ratio reference (the worker rebuilds its own oracle).
        built = case.build(param)
        attempts: List[AttemptRecord] = []
        for name, (framework, _fn) in built.baselines.items():
            spec = dict(
                run_id=run_id, case=case.name, param_point=param,
                baseline=name, platform=platform, warmup=warmup,
                repeats=repeats,
            )
            attempts.append(
                _spawn_worker(spec, platform, framework=framework,
                              timeout=timeout)
            )
        attach_ratios(attempts, built.ratio_reference)
        records.extend(attempts)
    return records


# --------------------------------------------------------------------------- #
# In-process driver (P0 fallback; memory metrics are process HWMs)            #
# --------------------------------------------------------------------------- #
def run_case_inprocess(
    case: Any, *, platform: str, warmup: int, repeats: int,
    prov: Dict[str, Any], run_id: str,
) -> List[AttemptRecord]:
    records: List[AttemptRecord] = []
    for param in case.param_points:
        built = case.build(param)
        attempts = [
            measure_attempt(
                case, param, built, name, platform=platform, run_id=run_id,
                prov=prov, warmup=warmup, repeats=repeats,
            )
            for name in built.baselines
        ]
        attach_ratios(attempts, built.ratio_reference)
        records.extend(attempts)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='semiring_matmul', choices=sorted(CASES))
    ap.add_argument('--platform', default='jax-cpu',
                    choices=['jax-cpu', 'jax-cuda12'],
                    help='target env-group for subprocess workers')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--quick', action='store_true',
                    help='representative param point only')
    ap.add_argument('--point', default=None, metavar='JSON',
                    help='run a single explicit param point (JSON)')
    ap.add_argument('--in-process', action='store_true',
                    help='P0 in-process driver (no spawn; memory = proc HWM)')
    ap.add_argument('--worker-timeout', type=float, default=3600.0,
                    help='per-attempt subprocess timeout (s)')
    ap.add_argument('--out', default=None,
                    help='default: results/<case>.jsonl')
    ap.add_argument('--report', default=None,
                    help='default: results/<case>.md')
    ap.add_argument('--render-from', default=None, metavar='JSONL',
                    help='re-render a report from saved L4 rows and exit')
    args = ap.parse_args()

    if args.render_from is not None:
        rows = read_jsonl(args.render_from)
        if not rows:
            raise SystemExit(f'no rows in {args.render_from}')
        report = render_markdown(rows, rows[0].get('provenance', {}))
        report_path = args.report or f'results/{rows[0].get("case")}.md'
        Path(report_path).write_text(report)
        print(report)
        print(f'Re-rendered {len(rows)} rows -> {report_path}.')
        return

    case = CASES[args.case]
    out_path_arg = args.out or f'results/{case.name}.jsonl'
    report_path_arg = args.report or f'results/{case.name}.md'
    if args.point is not None:
        case = replace(case, param_points=[json.loads(args.point)])
    elif args.quick:
        case = replace(case, param_points=[case.representative])

    if args.in_process:
        # The orchestrator *is* the measurer here; it needs the target backend.
        import jax
        jax.config.update('jax_default_matmul_precision', 'highest')
        jax.config.update('jax_enable_x64', True)
        prov = capture()
        prov['measurement_isolation'] = 'in_process'
        run_id = make_run_id(prov)
        records = run_case_inprocess(
            case, platform=platform_from(prov), warmup=args.warmup,
            repeats=args.repeats, prov=prov, run_id=run_id,
        )
    else:
        # Orchestrator only coordinates + builds oracles on CPU; the workers
        # own the target device.  Pin CPU via jax.config (not os.environ: jax
        # already snapshotted JAX_PLATFORMS at import) so oracle builds /
        # run_id never need the (possibly absent) target backend.  Workers get
        # their own backend explicitly via _worker_env.
        import jax
        jax.config.update('jax_platforms', 'cpu')
        run_id = make_run_id(capture())
        records = run_case_subprocess(
            case, platform=args.platform, warmup=args.warmup,
            repeats=args.repeats, run_id=run_id, timeout=args.worker_timeout,
        )

    # Report provenance = the rows' own (worker-captured authoritative device).
    report_prov = records[0].provenance if records else {}
    out_path = write_jsonl(records, out_path_arg)
    report = render_markdown([r.to_json() for r in records], report_prov)
    Path(report_path_arg).write_text(report)
    print(report)
    n_ok = sum(r.status == Status.OK for r in records)
    print(f'{len(records)} attempts ({n_ok} ok). Wrote {out_path} and '
          f'{report_path_arg}.')


if __name__ == '__main__':
    main()
