# -*- coding: utf-8 -*-
"""P0a driver (a minimal in-process L3 runner).

Runs one case across its baselines/param points and writes L4 rows + a
rendered report.  This is *not* the full runner — subprocess workers per env,
the device lock, and the sweep matrix are P1/P3.  What P0a exercises is the
per-attempt lifecycle that the schema depends on: ``jax.clear_caches()``
between attempts (cold compile, annex §D), the oracle computed *once per param
point* and shared across baselines (annex §C), exception → ``status`` row
(failure is data, never fatal), and the fidelity gate → *refuse the ratio but
record the absolutes* (DESIGN §1).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import jax
import numpy as np

from .cases import Case, semiring_matmul, throwaway
from .core import (
    AttemptRecord,
    Status,
    bench_call,
    capture,
    compare,
    host_rss_mb,
    make_run_id,
    peak_hbm_mb,
    read_jsonl,
    write_jsonl,
)
from .core.sync import SYNC
from .report import render_markdown

CASES: Dict[str, Case] = {
    throwaway.CASE.name: throwaway.CASE,
    semiring_matmul.CASE.name: semiring_matmul.CASE,
}


def _platform_from(prov: Dict[str, Any]) -> str:
    backend = (prov.get('device') or {}).get('platform')
    return {'gpu': 'jax-cuda12', 'cpu': 'jax-cpu'}.get(
        backend, f'jax-{backend}'
    )


def _classify_exception(exc: Exception) -> Tuple[Status, Dict[str, Any]]:
    msg = f'{type(exc).__name__}: {exc}'
    low = msg.lower()
    if 'resource_exhausted' in low or 'out of memory' in low:
        return Status.OOM, {'message': msg}
    # Requested backend / device simply isn't present on this host (e.g. the
    # pallas-cuda kernel on a CPU box).  The env imported fine and nothing
    # failed to compile -- the hardware is absent -- so this is a recorded
    # *skip*, not a compile_error.  Message-heuristic, same style as OOM above.
    if 'visible' in low and ('gpu' in low or 'device' in low):
        return Status.SKIPPED, {
            'reason': 'backend_unavailable', 'message': msg,
        }
    return Status.COMPILE_ERROR, {'message': msg}


def _measure_one(
    case: Case,
    param: Dict[str, Any],
    name: str,
    framework: str,
    fn: Callable[..., Any],
    inputs_for: Callable[[str], Tuple[Any, ...]],
    fp64_reference: Any,
    *,
    platform: str,
    run_id: str,
    prov: Dict[str, Any],
    warmup: int,
    repeats: int,
) -> AttemptRecord:
    base = dict(
        run_id=run_id, case=case.name, param_point=param, baseline=name,
        platform=platform, framework=framework, provenance=prov,
    )
    try:
        jax.clear_caches()  # cold compile per attempt (annex §D)
        args = inputs_for(framework)
        run_fn = jax.jit(fn) if framework == 'jax' else fn
        sync = SYNC[framework]
        compile_s, dist = bench_call(
            run_fn, args, warmup=warmup, repeats=repeats, sync=sync,
        )
        out = run_fn(*args)
        sync(out)
        out_host = np.asarray(out, dtype=np.float64)
        fid = compare(
            out_host, fp64_reference, rtol=case.rtol, atol=case.atol,
        )
        metrics = {
            'steady_time': {**dist.summary(), 'unit': 's'},
            'compile_time': {'value': compile_s, 'unit': 's', 'cache': 'cold'},
            'peak_hbm': {'value': peak_hbm_mb(), 'unit': 'MB'},
            'host_rss': {'value': host_rss_mb(), 'unit': 'MB'},
            'throughput': {
                'value': float(out_host.size) / dist.min, 'unit': 'elem/s',
            },
        }
        if fid['status'] == 'pass':
            return AttemptRecord(
                **base, status=Status.OK, metrics=metrics, fidelity=fid,
            )
        # Refuse the ratio, but keep the absolutes + the failing record.
        return AttemptRecord(
            **base, status=Status.FIDELITY_FAILED, metrics=metrics,
            fidelity=fid, failure_detail={'fidelity': fid},
        )
    except Exception as exc:  # noqa: BLE001 -- failure is data; classified.
        status, detail = _classify_exception(exc)
        return AttemptRecord(**base, status=status, failure_detail=detail)


def _attach_ratios(
    attempts: List[AttemptRecord], ratio_reference: str
) -> None:
    ref = next(
        (a for a in attempts
         if a.baseline == ratio_reference and a.status == Status.OK), None,
    )
    if ref is None or not ref.metrics:
        return
    ref_min = ref.metrics['steady_time']['min']
    for a in attempts:
        if a.status == Status.OK and a.metrics:
            a.ratio = {
                'vs': ratio_reference, 'metric': 'min',
                'value': a.metrics['steady_time']['min'] / ref_min,
            }


def run_case(
    case: Case,
    *,
    platform: str,
    warmup: int,
    repeats: int,
    prov: Dict[str, Any],
    run_id: str,
) -> List[AttemptRecord]:
    records: List[AttemptRecord] = []
    for param in case.param_points:
        built = case.build(param)
        attempts = [
            _measure_one(
                case, param, name, fw, fn, built.inputs_for,
                built.fp64_reference, platform=platform, run_id=run_id,
                prov=prov, warmup=warmup, repeats=repeats,
            )
            for name, (fw, fn) in built.baselines.items()
        ]
        _attach_ratios(attempts, built.ratio_reference)
        records.extend(attempts)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='semiring_matmul', choices=sorted(CASES))
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--quick', action='store_true',
                    help='representative param point only')
    ap.add_argument('--out', default=None,
                    help='default: results/<case>.jsonl')
    ap.add_argument('--report', default=None,
                    help='default: results/<case>.md')
    ap.add_argument('--render-from', default=None, metavar='JSONL',
                    help='re-render a report from saved L4 rows and exit '
                         '(no measurement); pairs with --report')
    args = ap.parse_args()

    # Re-render path: derive the report straight from stored rows (the renderer
    # does no arithmetic, so a saved run re-renders identically).  Lets a
    # renderer change be applied without re-running an expensive sweep.
    if args.render_from is not None:
        rows = read_jsonl(args.render_from)
        if not rows:
            raise SystemExit(f'no rows in {args.render_from}')
        prov = rows[0].get('provenance', {})
        report = render_markdown(rows, prov)
        report_path = args.report or f'results/{rows[0].get("case")}.md'
        Path(report_path).write_text(report)
        print(report)
        print(f'Re-rendered {len(rows)} rows -> {report_path}.')
        return

    # Fair, *recorded* numeric policy (DESIGN §L2 fairness / annex §C):
    #  - matmul precision 'highest' = true fp32, so GPU tensor cores don't
    #    silently downgrade to TF32 (which the fp64 oracle catches as a
    #    fidelity failure on a tensor-core GPU);
    #  - x64 enabled so the fp64 oracle is *actually* fp64 (else jax downcasts
    #    it to fp32 and the fidelity number is fp32-vs-fp32, not vs-truth).
    # Both are captured into provenance.  Per-baseline precision (a deliberate
    # TF32 baseline) is a P1 item.
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_enable_x64', True)

    case = CASES[args.case]
    out_path_arg = args.out or f'results/{case.name}.jsonl'
    report_path_arg = args.report or f'results/{case.name}.md'
    if args.quick:
        case = replace(case, param_points=[case.representative])

    prov = capture()
    run_id = make_run_id(prov)
    platform = _platform_from(prov)
    records = run_case(
        case, platform=platform, warmup=args.warmup, repeats=args.repeats,
        prov=prov, run_id=run_id,
    )
    out_path = write_jsonl(records, out_path_arg)
    report = render_markdown([r.to_json() for r in records], prov)
    Path(report_path_arg).write_text(report)
    print(report)
    n_ok = sum(r.status == Status.OK for r in records)
    print(f'{len(records)} attempts ({n_ok} ok). Wrote {out_path} and '
          f'{report_path_arg}.')


if __name__ == '__main__':
    main()
