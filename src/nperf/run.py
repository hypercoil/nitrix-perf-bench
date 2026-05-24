# -*- coding: utf-8 -*-
"""Runner / orchestrator (L3).

Default mode (**subprocess**): one OS process per attempt, spawned via a
pluggable interpreter (`worker.py`).  This is the P1 runner — it makes
per-attempt memory honest (each worker's high-water mark *is* its attempt's
peak; see `core/memory.py`) and isolates crashes (a dead worker becomes a
failure row, the sweep continues).  Attempts fan across the requested
`--platforms` and across `--gpus` devices, scheduled by `schedule.py`: distinct
resources (CPU vs a GPU, or two GPUs) overlap, while a per-device lock / pinned
CPU slots serialise within a resource (annex §E).

`--in-process` keeps the P0 driver (no spawn cost, faster for CPU smoke) — but
its memory metrics are process high-water marks, so the renderer flags them.

`--store` ingests a run into the durable per-run store (`store.py`);
`--render-from <files/dirs>` re-renders (and combines) saved L4 rows — no
measurement — with `--latest` to collapse to current state per key.
`--gate-baseline <files/dirs>` enters **gate mode** (`gate.py`, SCHEMA §F): no
measurement — diff `--gate-current` (default: the store) against the baseline
on `steady_time` min + p95, write the artifact + markdown diff, and exit
nonzero if any key regressed (a CI check).

Worker interpreter resolution (per *attempt*, by framework + platform): a
framework the base env doesn't ship (torch, pyg) resolves
`NPERF_PYTHON_<FW>_<PLATFORM>` then `NPERF_PYTHON_<FW>` first (its own isolated
env, e.g. `NPERF_PYTHON_TORCH` from tools/setup_refs_env.sh); jax/numpy skip
straight to the platform-wide `NPERF_PYTHON_<PLATFORM>`, then
`NPERF_WORKER_PYTHON`, else this interpreter (the uv default when you run the
orchestrator under the matching env; for a platform this env can't provide,
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
from typing import Any, Dict, List, Optional

from . import gate, store
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
from .providers import framework_of
from .report import render_gate, render_markdown, render_site
from .schedule import ResourcePool, resource_of, run_scheduled

_SRC = str(Path(__file__).resolve().parents[1])  # `src` dir, for PYTHONPATH

# Frameworks the orchestrator / base env already ships, so their workers run
# under the platform interpreter (the historical behaviour).  Everything else
# (torch, pyg) is an isolated env that needs its own interpreter even on a jax
# platform's device -- hence the framework-specific overrides below.
_BASE_FRAMEWORKS = frozenset({'jax', 'numpy'})


# --------------------------------------------------------------------------- #
# Subprocess orchestration                                                    #
# --------------------------------------------------------------------------- #
def _worker_python(platform: str, framework: str = 'jax') -> str:
    '''Resolve the worker interpreter for a ``(framework, platform)`` attempt.

    A framework the orchestrator env does not ship (torch, pyg) needs its
    *own* interpreter even when the target *device* matches a jax
    platform -- so its framework-specific overrides win first
    (``NPERF_PYTHON_<FW>_<PLATFORM>`` then ``NPERF_PYTHON_<FW>``); then the
    platform-wide override (``NPERF_PYTHON_<PLATFORM>`` -- back-compat for the
    jax/numpy workers, which only ever consult this); then the global
    ``NPERF_WORKER_PYTHON``; else this interpreter.'''
    keys: List[str] = []
    plat = platform.upper().replace('-', '_')
    if framework not in _BASE_FRAMEWORKS:
        fw = framework.upper()
        keys += ['NPERF_PYTHON_%s_%s' % (fw, plat), 'NPERF_PYTHON_' + fw]
    keys += ['NPERF_PYTHON_' + plat, 'NPERF_WORKER_PYTHON']
    override = next((os.environ[k] for k in keys if os.environ.get(k)), None)
    return override or sys.executable


def _worker_argv(platform: str, framework: str = 'jax') -> List[str]:
    return [_worker_python(platform, framework), '-m', 'nperf.worker']


def _worker_env(
    platform: str, cores: Optional[List[int]] = None,
    device: Optional[int] = None,
) -> Dict[str, str]:
    env = dict(os.environ)
    # nperf must be importable even when the interpreter is a prebuilt env that
    # never pip-installed it (nitrix is already importable in those by design).
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = _SRC + (os.pathsep + existing if existing else '')
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # honest memory_stats
    # Pin the worker's backend explicitly per target platform.
    env['JAX_PLATFORMS'] = 'cuda' if platform == 'jax-cuda12' else 'cpu'
    # CPU slot's disjoint core group -> the worker pins to it pre jax-import.
    if cores:
        env['NPERF_CPU_CORES'] = ','.join(str(c) for c in cores)
    # GPU attempts are pinned to their assigned physical device (multi-GPU
    # fan-out): the worker then sees exactly that one device as cuda:0.
    if device is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(device)
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
    spec: Dict[str, Any], ctx: Dict[str, Any], *, timeout: float
) -> AttemptRecord:
    '''Run one attempt in a fresh worker process (scheduler `run_one`).

    ``ctx`` is the resource context from the pool: ``ctx["cores"]`` is this
    attempt's pinned CPU core group (None on GPU); ``ctx["device"]`` is its
    assigned physical GPU id (None on CPU).'''
    platform = spec['platform']
    base = dict(
        run_id=spec['run_id'], case=spec['case'],
        param_point=spec['param_point'], baseline=spec['baseline'],
        platform=platform, framework=spec['framework'], provenance={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / 'row.jsonl'
        spec_path = Path(tmp) / 'spec.json'
        spec_path.write_text(
            json.dumps({**spec, 'result_path': str(result_path)})
        )
        argv = (_worker_argv(platform, spec['framework'])
                + ['--spec', str(spec_path)])
        try:
            proc = subprocess.run(
                argv,
                env=_worker_env(platform, ctx.get('cores'), ctx.get('device')),
                capture_output=True, text=True, timeout=timeout,
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


def _probe_gpu_count(platform: str = 'jax-cuda12') -> int:
    '''Ask the GPU worker interpreter how many devices it sees (>=1).

    The orchestrator runs CPU-pinned, so it can't enumerate GPUs itself; a tiny
    probe in the worker env does.  Any failure -> 1 (assume one device).'''
    argv = [_worker_argv(platform)[0], '-c',
            'import jax; print(len(jax.devices("gpu")))']
    try:
        proc = subprocess.run(
            argv, env=_worker_env(platform), capture_output=True, text=True,
            timeout=120,
        )
        return max(1, int(proc.stdout.strip().splitlines()[-1]))
    except Exception:
        return 1


def run_case_subprocess(
    case: Any, *, platforms: List[str], warmup: int, repeats: int,
    run_id: str, timeout: float, cpu_slots: int, max_parallel: Optional[int],
    gpu_settle_s: float, n_gpus: int = 1,
) -> List[AttemptRecord]:
    # Fan attempts across the requested platforms: for each param point (oracle
    # built once, shared) and each platform, emit the platform's baselines as a
    # contiguous group tagged with the physical resource it contends for.  The
    # scheduler then overlaps distinct resources (e.g. CPU and a GPU) while the
    # device lock / CPU slots serialise within a resource.
    specs: List[Dict[str, Any]] = []
    groups: List[tuple] = []  # (start, n_baselines, ratio_reference)
    for param in case.param_points:
        built = case.build(param)
        names = list(built.baselines)
        for platform in platforms:
            resource = resource_of(platform)
            start = len(specs)
            for name in names:
                specs.append(dict(
                    run_id=run_id, case=case.name, param_point=param,
                    baseline=name, platform=platform, warmup=warmup,
                    repeats=repeats,
                    framework=framework_of(built.baselines[name][0]),
                    resource=resource,
                ))
            groups.append((start, len(names), built.ratio_reference))

    resources = {resource_of(p) for p in platforms}
    gpu_present = 'gpu' in resources
    pool = ResourcePool(
        cpu_slots=cpu_slots, n_gpus=(n_gpus if gpu_present else 0),
        gpu_settle_s=gpu_settle_s,
    )
    # Default: enough threads to keep every resource busy at once (cpu slots +
    # every GPU).
    cpu_permits = pool.cpu_slots if 'cpu' in resources else 0
    gpu_permits = pool.n_gpus if gpu_present else 0
    n_parallel = max_parallel or max(1, cpu_permits + gpu_permits)
    sched = {
        'cpu_slots': pool.cpu_slots, 'n_gpus': pool.n_gpus,
        'max_parallel': n_parallel, 'core_groups': pool.core_groups,
        'gpu_settle_s': gpu_settle_s,
    }
    records: List[AttemptRecord] = run_scheduled(
        specs, lambda s, ctx: _spawn_worker(s, ctx, timeout=timeout),
        pool, max_parallel=n_parallel,
    )

    # Ratios are computed *within* each (param point, platform) group -- a GPU
    # kernel must be rated against its platform's own reference, never across
    # platforms.  Then stamp the scheduler regime onto every row's provenance.
    for start, n_baselines, ratio_reference in groups:
        attach_ratios(records[start:start + n_baselines], ratio_reference)
    for rec in records:
        rec.provenance['scheduler'] = sched
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


def _run_gate(args: Any) -> None:
    '''Gate mode (SCHEMA §F): compare current rows against a baseline, write a
    machine-readable artifact + a markdown diff, and exit **nonzero** if any
    key regressed (so CI fails the PR).  Inputs are .jsonl files or store dirs,
    combined like ``--render-from``; current defaults to the whole store.'''
    def _read(paths: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for f in store.expand_inputs(paths):
            rows.extend(read_jsonl(f))
        return rows

    current_inputs = args.gate_current or [store.STORE_DEFAULT]
    base_rows = _read(args.gate_baseline)
    curr_rows = _read(current_inputs)
    if not base_rows:
        raise SystemExit(f'gate: no baseline rows in {args.gate_baseline}')
    if not curr_rows:
        raise SystemExit(f'gate: no current rows in {current_inputs}')
    artifact = gate.compare(
        base_rows, curr_rows,
        min_threshold=args.gate_min, p95_threshold=args.gate_p95,
    )
    out = Path(args.gate_out or 'results/gate.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2) + '\n')
    report_path = Path(args.report or 'results/gate.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_gate(artifact)
    report_path.write_text(md)
    print(md)
    s = artifact['summary']
    print(f"Gate {s['verdict'].upper()}: {s['n_regressed']} regressed of "
          f"{s['n_compared']} compared. Artifact {out}, report {report_path}.")
    raise SystemExit(0 if s['verdict'] == 'pass' else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='semiring_matmul', choices=sorted(CASES))
    ap.add_argument('--platforms', default='jax-cpu',
                    help='comma-list of target env-groups for subprocess '
                         'workers (e.g. "jax-cpu,jax-cuda12"); attempts fan '
                         'out across them and distinct resources run in '
                         'parallel')
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--quick', action='store_true',
                    help='representative param point only')
    ap.add_argument('--point', default=None, metavar='JSON',
                    help='run a single explicit param point (JSON)')
    ap.add_argument('--in-process', action='store_true',
                    help='P0 in-process driver (no spawn; memory = proc HWM)')
    ap.add_argument('--cpu-slots', type=int, default=1,
                    help='parallel CPU attempts, each pinned to a disjoint '
                         'core group (1 = serial; >1 trades nothing on timing '
                         'fidelity because slots are core-disjoint)')
    ap.add_argument('--gpus', type=int, default=None,
                    help='physical GPUs to fan attempts across, one device '
                         'lock each (default: auto-probe the GPU worker '
                         'interpreter; 1 if none)')
    ap.add_argument('--max-parallel', type=int, default=None,
                    help='cap concurrent workers (default: resource permits: '
                         'cpu_slots + every GPU)')
    ap.add_argument('--gpu-settle', type=float, default=0.0,
                    help='seconds held inside the GPU device lock between '
                         'attempts (clock-state settle)')
    ap.add_argument('--worker-timeout', type=float, default=3600.0,
                    help='per-attempt subprocess timeout (s)')
    ap.add_argument('--out', default=None,
                    help='default: results/<case>.jsonl')
    ap.add_argument('--report', default=None,
                    help='default: results/<case>.md')
    ap.add_argument('--render-from', default=None, nargs='+', metavar='PATH',
                    help='re-render a report from saved L4 rows and exit; '
                         'each PATH is a .jsonl file or a directory (a store '
                         'root / case dir, globbed) -- combines runs/devices '
                         'into one multi-platform report (DESIGN §8)')
    ap.add_argument('--latest', action='store_true',
                    help='with --render-from: keep only the newest row per '
                         '(case, platform, param, baseline) across runs')
    ap.add_argument('--site', nargs='?', const='site', default=None,
                    metavar='DIR',
                    help='render the self-contained HTML /site (default dir: '
                         'site) from --render-from rows (or the whole store); '
                         'tables show current state, plots include history; '
                         'git-ignored')
    ap.add_argument('--op-matrix', default=None, metavar='PATH',
                    help="with --site: overlay nitrix's capability op_matrix "
                         '.json as the overview (default: '
                         '../nitrix/docs/op_matrix.json if present)')
    ap.add_argument('--store', nargs='?', const=store.STORE_DEFAULT,
                    default=None, metavar='DIR',
                    help='also ingest this run durably into the store '
                         f'(default DIR: {store.STORE_DEFAULT})')
    ap.add_argument('--prune-keep', type=int, default=None, metavar='N',
                    help='with --store: keep only the N most recent runs')
    # Regression gate (mode; SCHEMA §F). --gate-baseline triggers it.
    ap.add_argument('--gate-baseline', nargs='+', default=None, metavar='PATH',
                    help='regression gate: stored baseline rows (.jsonl files '
                         'or store dirs); presence selects gate mode and '
                         'exits nonzero if the current run regressed')
    ap.add_argument('--gate-current', nargs='+', default=None, metavar='PATH',
                    help='gate: current rows to test (default: the store)')
    ap.add_argument('--gate-min', type=float,
                    default=gate.DEFAULT_MIN_THRESHOLD, metavar='X',
                    help='gate: tight min-ratio threshold (default '
                         f'{gate.DEFAULT_MIN_THRESHOLD})')
    ap.add_argument('--gate-p95', type=float,
                    default=gate.DEFAULT_P95_THRESHOLD, metavar='X',
                    help='gate: loose p95-ratio threshold (default '
                         f'{gate.DEFAULT_P95_THRESHOLD})')
    ap.add_argument('--gate-out', default=None, metavar='PATH',
                    help='gate: machine-readable artifact (default '
                         'results/gate.json)')
    args = ap.parse_args()

    if args.gate_baseline is not None:
        _run_gate(args)
        return

    if args.site is not None:
        # Site mode: read rows (the explicit --render-from set, else the whole
        # store) WITHOUT collapsing -- render_site keeps current state for the
        # tables and uses every run for the history plots.
        inputs = args.render_from or [store.STORE_DEFAULT]
        site_rows: List[Dict[str, Any]] = []
        for path in store.expand_inputs(inputs):
            site_rows.extend(read_jsonl(path))
        if not site_rows:
            raise SystemExit(f'no rows for --site in {inputs}')
        # Overlay nitrix's capability matrix (capability stays nitrix's; perf
        # is ours).  Default to the sibling checkout if present; absent -> the
        # site just omits the capability section.
        cap_path = Path(
            args.op_matrix
            or (_SRC and Path(_SRC).parent.parent / 'nitrix' / 'docs'
                / 'op_matrix.json')
        )
        capability = (json.loads(cap_path.read_text())
                      if cap_path.exists() else None)
        site_dir = Path(args.site)
        site_dir.mkdir(parents=True, exist_ok=True)
        index = site_dir / 'index.html'
        index.write_text(render_site(site_rows, capability=capability))
        cap_note = '' if capability is None else ' + nitrix capability matrix'
        print(f'Wrote site -> {index} ({len(site_rows)} rows from '
              f'{len(store.expand_inputs(inputs))} file(s){cap_note}).')
        return

    if args.render_from is not None:
        files = store.expand_inputs(args.render_from)
        rows: List[Dict[str, Any]] = []
        for path in files:
            rows.extend(read_jsonl(path))
        if args.latest:
            rows = store.latest(rows)
        if not rows:
            raise SystemExit(f'no rows in {args.render_from}')
        report = render_markdown(rows, rows[0].get('provenance', {}))
        report_path = args.report or f'results/{rows[0].get("case")}.md'
        Path(report_path).write_text(report)
        print(report)
        print(f'Re-rendered {len(rows)} rows from {len(files)} file(s) -> '
              f'{report_path}.')
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
        platforms = [p.strip() for p in args.platforms.split(',') if p.strip()]
        gpu_present = any(resource_of(p) == 'gpu' for p in platforms)
        n_gpus = args.gpus
        if n_gpus is None:
            n_gpus = _probe_gpu_count() if gpu_present else 1
        records = run_case_subprocess(
            case, platforms=platforms, warmup=args.warmup,
            repeats=args.repeats, run_id=run_id, timeout=args.worker_timeout,
            cpu_slots=args.cpu_slots, max_parallel=args.max_parallel,
            gpu_settle_s=args.gpu_settle, n_gpus=n_gpus,
        )

    # Report provenance = the rows' own (worker-captured authoritative device).
    report_prov = records[0].provenance if records else {}
    out_path = write_jsonl(records, out_path_arg)
    report = render_markdown([r.to_json() for r in records], report_prov)
    Path(report_path_arg).write_text(report)
    print(report)
    n_ok = sum(r.status == Status.OK for r in records)
    msg = (f'{len(records)} attempts ({n_ok} ok). Wrote {out_path} and '
           f'{report_path_arg}.')

    # Durable accumulation: append this run to the store (one file per run),
    # then optionally cap history.  Combine later with `--render-from <store>`.
    if args.store is not None:
        stored = store.ingest(
            records, root=args.store, case=case.name, run_id=run_id,
        )
        msg += f' Ingested -> {stored}.'
        if args.prune_keep is not None:
            removed = store.prune(args.store, case.name, args.prune_keep)
            msg += f' Pruned {len(removed)} old run(s).'
    print(msg)


if __name__ == '__main__':
    main()
