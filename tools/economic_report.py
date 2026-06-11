# -*- coding: utf-8 -*-
"""Economic verdict: is the nitrix-GPU win *multiplicative* over the CPU gold
standard -- enough to be worth the GPU hardware premium for a real user?

The principle (the registration framing, but general): an *incremental*
nitrix-GPU win over the ANTs / ITK / numpy CPU gold standard is **not** a win
for a practitioner, because a GPU-hour costs several times a CPU-hour.  A win
only counts when it is **multiplicative** -- it clears an explicit GPU:CPU
cost-multiple bar (``--cost-multiple``, default ~4x; the rendered header
states the cloud-pricing rationale).

This is a pure **store read** -- a deliberate *cross-platform* join the scaling
report (single-platform by charter) does not do: for each ``(case,
param_point)`` it pairs the ``nitrix-jax`` row on ``jax-cuda12`` (GPU steady +
the one-time GPU compile) with the fastest CPU **domain** baseline on
``jax-cpu`` (the strongest competitor; ANTs / dipy / numpy ...), falling back
to ``nitrix-jax`` on ``jax-cpu`` (GPU-vs-own-CPU) when the op has no domain
tool (e.g. BBR).  No new measurement, no schema change.

Two reads per point (the user's operationalisation):

- **amortized** = CPU walltime / nitrix-GPU **steady** (compile amortised over
  many subjects / frames -- the cohort / batched case);
- **single-run** = CPU walltime / (nitrix-GPU steady + **GPU compile**) -- one
  registration, paid cold.

Reads the store only (no GPU / measurement)::

    JAX_PLATFORMS=cpu python tools/economic_report.py
    JAX_PLATFORMS=cpu python tools/economic_report.py --cost-multiple 8
    JAX_PLATFORMS=cpu python tools/economic_report.py --case volreg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scaling_report import _label, _size_elems  # noqa: E402

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402

# GPU:CPU cost multiple -- an L4 GPU-hour costs ~4x an equivalent CPU-hour on
# the major clouds (an L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs
# a comparable general-purpose vCPU instance ~$0.18/hr, 2026). A nitrix-GPU
# result is only "economically favorable" if it beats the CPU gold standard by
# MORE than this -- an incremental GPU win is not a win once the user pays the
# GPU premium. Tunable via --cost-multiple.
_GPU_CPU_COST_MULTIPLE = 4.0
_GPU = 'jax-cuda12'
_CPU = 'jax-cpu'


def _steady_min(row: Dict[str, Any]) -> Optional[float]:
    return ((row.get('metrics') or {}).get('steady_time') or {}).get('min')


def _compile(row: Dict[str, Any]) -> Optional[float]:
    return ((row.get('metrics') or {}).get('compile_time') or {}).get('value')


def _pkey(param: Dict[str, Any]) -> str:
    return json.dumps(param, sort_keys=True)


def _verdict(amortized: Optional[float], single: Optional[float],
             bar: float) -> str:
    '''The economic label for one point at the cost-multiple ``bar``.'''
    if amortized is None:
        return 'n/a'
    if amortized < bar:
        # a real GPU win can still be NOT a win once the hardware premium is
        # paid -- the user's central point.
        return 'not multiplicative enough'
    if single is not None and single >= bar:
        return 'favorable'
    return 'favorable (amortized only)'  # the compile is the gate


def _analyse(case: Any, rows: List[dict], bar: float) -> List[Dict[str, Any]]:
    '''Join GPU-nitrix x CPU-gold per param point for one case.

    Restricted to the **size tier** (``large_param_points``): that is where the
    scale-relevant comparison lives, and -- crucially -- the CPU domain tools
    (ANTs/dipy) run a FIXED schedule that ignores the dev ``(levels, iters)``
    configs, so a verdict on those would be meaningless.'''
    large_keys = {_pkey(p) for p in case.large_param_points}
    if not large_keys:
        return []
    by_key: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get('case') != case.name:
            continue
        k = _pkey(r['param_point'])
        if k in large_keys:
            by_key.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for grp in by_key.values():
        gpu = [r for r in grp if r['baseline'] == 'nitrix-jax'
               and r['platform'] == _GPU and r['status'] == 'ok']
        if not gpu or _steady_min(gpu[0]) is None:
            continue
        gs, gc = _steady_min(gpu[0]), _compile(gpu[0])
        cpu_ok = [r for r in grp if r['platform'] == _CPU
                  and r['status'] == 'ok' and _steady_min(r) is not None]
        # I/O floors by provider namespace (the name prefix before the dot):
        # a CLI tool's wall-clock includes a NIfTI write+subprocess+read the
        # in-memory nitrix op never pays -- subtract the same-namespace no-op
        # (afni.iofloor / fsl.iofloor) to isolate the registration COMPUTE.
        floors = {r['baseline'].split('.')[0]: _steady_min(r)
                  for r in cpu_ok if r['baseline'].endswith('.iofloor')}

        def _compute(r: dict) -> float:
            f = floors.get(r['baseline'].split('.')[0], 0.0)
            return max(_steady_min(r) - f, 1e-6)  # I/O-subtracted

        domain = [r for r in cpu_ok if r['baseline'] != 'nitrix-jax'
                  and not r['baseline'].endswith('.iofloor')]
        if domain:  # strongest competitor = fastest tool AFTER I/O subtraction
            best = min(domain, key=_compute)
            tool, fallback = best['baseline'], False
            cpu_raw = _steady_min(best)
            iofloor = floors.get(tool.split('.')[0])
            cpu = _compute(best)
        else:  # no domain tool (e.g. BBR) -> GPU-vs-own-CPU (no I/O artifact)
            nc = [r for r in cpu_ok if r['baseline'] == 'nitrix-jax']
            if not nc:
                continue
            tool, fallback = 'nitrix-CPU', True
            cpu_raw, iofloor = _steady_min(nc[0]), None
            cpu = cpu_raw
        amort = cpu / gs if gs and gs > 0 else None
        single = (cpu / (gs + gc)) if (gc is not None and (gs + gc) > 0
                                       ) else None
        param = gpu[0]['param_point']
        out.append({
            'label': _label(param), 'size': _size_elems(param),
            'gpu_steady': gs, 'gpu_compile': gc, 'cpu': cpu, 'tool': tool,
            'cpu_raw': cpu_raw, 'iofloor': iofloor,
            'fallback': fallback, 'amort': amort, 'single': single,
            'verdict': _verdict(amort, single, bar),
        })
    out.sort(key=lambda d: d['size'])
    return out


def _t(s: Optional[float]) -> str:
    if s is None:
        return '—'
    return f'{s * 1e3:.1f} ms' if s < 1 else f'{s:.2f} s'


def _x(v: Optional[float]) -> str:
    return '—' if v is None else f'{v:.1f}x'


def _render(case: Any, rows: List[Dict[str, Any]], bar: float) -> List[str]:
    doc = [f'## {case.name}  ({case.op_qualname})', '']
    if any(r['fallback'] for r in rows):
        doc.append('> No CPU **domain** tool for this op -- the CPU bar is '
                   'nitrix-CPU (GPU-vs-own-CPU: is the GPU worth the premium '
                   'for *this* op).')
        doc.append('')
    if any(r.get('iofloor') is not None for r in rows):
        doc.append('> CPU times for the CLI tools (AFNI/FSL) are **I/O-'
                   'subtracted**: `compute = tool wall-clock - the matching '
                   'no-op` (`3dcalc`/`fslmaths` identity = the NIfTI '
                   'round-trip nitrix never pays). Raw and floor shown in the '
                   'tool cell.')
        doc.append('')
    doc += ['| size | GPU steady | GPU compile | CPU compute (tool) | '
            'amortized | single-run | verdict |',
            '|---|---|---|---|---|---|---|']
    for r in rows:
        if r.get('iofloor') is not None:
            cell = (f'{_t(r["cpu"])} ({r["tool"]}; {_t(r["cpu_raw"])}'
                    f'−{_t(r["iofloor"])} io)')
        else:
            cell = f'{_t(r["cpu"])} ({r["tool"]})'
        doc.append(
            f'| {r["label"]} | {_t(r["gpu_steady"])} | {_t(r["gpu_compile"])} '
            f'| {cell} | {_x(r["amort"])} | '
            f'{_x(r["single"])} | {r["verdict"]} |')
    fav = [r for r in rows if r['verdict'].startswith('favorable')]
    best = max(rows, key=lambda r: r['amort'] or 0)
    doc += ['', f'- **{len(fav)}/{len(rows)}** size(s) favorable at '
            f'{bar:g}x; best amortized **{_x(best["amort"])}** at '
            f'`{best["label"]}` (single-run {_x(best["single"])}).', '']
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='L4 rows: .jsonl files or store dirs (default store)')
    ap.add_argument('--cost-multiple', type=float,
                    default=_GPU_CPU_COST_MULTIPLE,
                    help='GPU:CPU $/hr premium -- the win bar (default ~4x)')
    ap.add_argument('--case', nargs='+', default=None,
                    help='restrict to these cases (default: all with a join)')
    ap.add_argument('--out-md', default='reports/ECONOMIC.md')
    args = ap.parse_args()

    rows: list = []
    for f in store.expand_inputs(args.inputs):
        rows.extend(read_jsonl(f))
    rows = store.latest(rows)
    bar = args.cost_multiple

    names = set(args.case) if args.case else None
    doc = [
        '# Economic verdict: nitrix-GPU vs the CPU gold standard', '',
        f'A GPU-hour costs **~{bar:g}x** a CPU-hour on the major clouds (an '
        'L4 instance, e.g. AWS g6.xlarge, ~$0.80/hr on-demand vs a comparable '
        'general-purpose vCPU instance ~$0.18/hr, 2026). So a nitrix-GPU '
        'result is *economically favorable* only when it beats the CPU gold '
        f'standard by MORE than {bar:g}x -- an incremental GPU win is **not** '
        'a win once a real user pays the GPU premium. Tunable via '
        '`--cost-multiple`.', '',
        '- **amortized** = CPU walltime / nitrix-GPU steady (compile '
        'amortised over many subjects / frames).',
        '- **single-run** = CPU walltime / (nitrix-GPU steady + GPU compile) '
        '-- one run, cold.',
        '- **verdict**: `favorable` (both >= bar) / `favorable (amortized '
        'only)` (the compile is the gate -- amortise it over the cohort) / '
        '`not multiplicative enough` (a real GPU win, but < bar -- so NOT a '
        'win by the cost test).', '',
        '**Caveats (read with care):** the CPU domain tools (ANTs / dipy) run '
        'a FIXED internal schedule and ignore our `(levels, iters)`, so the '
        'verdict is meaningful across the **size / T tier**, not the dev '
        'configs; nitrix runs a fixed-iteration scan while ANTs / dipy '
        'early-exit on convergence (a wall-clock economic read, not a '
        'per-iteration claim); **time only** (HBM excluded -- cold peak is '
        'autotune-contaminated, see `SCALING.md`). For volreg the CPU bar is '
        'the **community realignment standard** -- AFNI `3dvolreg` / FSL '
        '`mcflirt` (fast, hand-optimised C), **I/O-floor-subtracted** '
        '(`compute = tool - the matching 3dcalc/fslmaths no-op`); ANTs '
        '`motion_correction` is kept only as a slow reference (timed out '
        'at T=500).', '',
    ]
    n_fav = 0
    for case in sorted(CASES.values(), key=lambda c: c.name):
        if names is not None and case.name not in names:
            continue
        analysed = _analyse(case, rows, bar)
        if not analysed:
            continue
        if any(r['verdict'].startswith('favorable') for r in analysed):
            n_fav += 1
        doc += _render(case, analysed, bar)

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text('\n'.join(doc) + '\n')
    print(f'economic: verdict at {bar:g}x cost-multiple; '
          f'{n_fav} case(s) with a favorable size. Wrote {args.out_md}.',
          file=sys.stderr)


if __name__ == '__main__':
    main()
