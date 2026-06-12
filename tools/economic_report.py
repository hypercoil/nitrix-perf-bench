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
report (single-platform by charter) does not do.  The join itself lives in
``nperf.report.economic`` (shared with the coverage matrix's economic axis);
this tool is the ECONOMIC.md renderer over it.

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
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402
from nperf.report.economic import (  # noqa: E402
    COST_MULTIPLE as _GPU_CPU_COST_MULTIPLE,
)
from nperf.report.economic import analyse as _analyse


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
