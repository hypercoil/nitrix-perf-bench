# -*- coding: utf-8 -*-
"""Export perf ratios into the shape nitrix's op_matrix wants (P2 feed).

perf-bench is the source of truth for *performance*; nitrix's
``docs/op_matrix.json`` carries per-op ``perf_{cpu,gpu}_{baseline,ratio}``
fields (currently ``null``).  This tool reads accumulated L4 rows and emits,
per op, those fields **at the case's representative point** -- the op_matrix
necessarily collapses a whole sweep to one figure per device; the full
multi-point / cross-framework picture stays in the perf-bench report.

Ratio convention (recorded in the emitted ``_meta`` so it can't be misread)::

    perf_<dev>_ratio = nitrix_primary.steady_min / reference.steady_min

so **< 1 means nitrix is faster** than the reference.  Default reference:
``naive-dense`` -- the universal "what you'd write without nitrix" baseline,
present on cpu + gpu and every algebra (use ``--reference torch-dense`` for the
cross-framework number instead).  The nitrix primary is the case's own ratio
reference (``nitrix-jax``), read from the rows.

This tool does **not** mutate nitrix.  It prints a JSON fragment + a human
summary; ``--apply path/to/op_matrix.json`` writes a *merged copy* to
``--out`` for review (the actual change to nitrix is that repo's own commit).

Run (point it at the same rows the report uses)::

    JAX_PLATFORMS=cpu python tools/op_matrix_feed.py \
        --from reports/semiring_matmul.jsonl reports/semiring_matmul_cpu.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402

# A case measures one nitrix op; the op_matrix is keyed by the public qualname.
CASE_QUALNAME: Dict[str, str] = {
    'semiring_matmul': 'nitrix.semiring.semiring_matmul',
}

# perf-bench platform -> op_matrix device axis.
PLATFORM_DEVICE = {'jax-cpu': 'cpu', 'jax-cuda12': 'gpu'}


def _same_point(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    '''Match a row's param point to the representative (shape + algebra).'''
    return all(a.get(k) == b.get(k) for k in ('m', 'k', 'n', 'algebra'))


def _ok_min(row: Dict[str, Any]) -> Optional[float]:
    if row.get('status') != 'ok':
        return None
    try:
        return float(row['metrics']['steady_time']['min'])
    except (KeyError, TypeError):
        return None


def _device_ratio(
    rows: List[Dict[str, Any]], reference: str
) -> Dict[str, Any]:
    '''nitrix-primary-vs-reference ratio for one device's representative rows.

    Returns ``{baseline, ratio}`` (ratio = primary.min / reference.min) or an
    explanatory ``{baseline: None, ratio: None, note}`` when either side is
    missing / not ok.'''
    # The primary is whatever the rows take ratios against (the case's
    # ``ratio_reference``); read it off any row that carries a ratio.
    primary = next(
        (r['ratio']['vs'] for r in rows if r.get('ratio')), 'nitrix-jax',
    )
    by_name = {r.get('baseline'): r for r in rows}
    p_min = _ok_min(by_name.get(primary, {}))
    r_min = _ok_min(by_name.get(reference, {}))
    if p_min is None:
        return {'baseline': None, 'ratio': None,
                'note': f'primary {primary!r} not ok at this point'}
    if r_min is None:
        return {'baseline': None, 'ratio': None,
                'note': f'reference {reference!r} not ok at this point'}
    return {'baseline': reference, 'ratio': round(p_min / r_min, 4),
            'primary': primary}


def build_fragment(
    rows: List[Dict[str, Any]], case: str, reference: str
) -> Dict[str, Any]:
    rep = CASES[case].representative
    rep_rows = [
        r for r in rows
        if r.get('case') == case and _same_point(r['param_point'], rep)
    ]
    entry: Dict[str, Any] = {
        '_meta': {
            'point': {k: rep[k] for k in ('m', 'k', 'n', 'algebra')},
            'reference': reference,
            'ratio_convention': 'primary.steady_min / reference.steady_min '
                                '(<1 = nitrix faster)',
            'source': 'nitrix-perf-bench',
        }
    }
    for platform, device in PLATFORM_DEVICE.items():
        plat_rows = [r for r in rep_rows if r.get('platform') == platform]
        if not plat_rows:
            continue
        res = _device_ratio(plat_rows, reference)
        entry[f'perf_{device}_baseline'] = res['baseline']
        entry[f'perf_{device}_ratio'] = res['ratio']
        if res.get('note'):
            entry['_meta'].setdefault('notes', {})[device] = res['note']
    return {CASE_QUALNAME[case]: entry}


def _apply(op_matrix_path: Path, fragment: Dict[str, Any]) -> Dict[str, Any]:
    '''Merge the fragment's ``perf_*`` fields into a copy of op_matrix.json.'''
    doc = json.loads(op_matrix_path.read_text())
    by_q = {op.get('qualname'): op for op in doc.get('ops', [])}
    for qual, entry in fragment.items():
        op = by_q.get(qual)
        if op is None:
            continue
        for field, val in entry.items():
            if field.startswith('perf_'):
                op[field] = val
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='semiring_matmul', choices=sorted(CASES))
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='L4 rows: .jsonl files or store dirs (default: the '
                         'store). Newest row per key is used.')
    ap.add_argument('--reference', default='naive-dense',
                    help='baseline the nitrix primary is rated against '
                         '(default naive-dense; e.g. torch-dense)')
    ap.add_argument('--apply', metavar='OP_MATRIX_JSON',
                    help='merge perf_* fields into a COPY of this op_matrix '
                         '(written to --out; never mutates nitrix in place)')
    ap.add_argument('--out', metavar='PATH',
                    help='with --apply: where to write the merged copy')
    args = ap.parse_args()

    files = store.expand_inputs(args.inputs)
    rows: List[Dict[str, Any]] = []
    for f in files:
        rows.extend(read_jsonl(f))
    rows = store.latest(rows)
    if not rows:
        raise SystemExit(f'no rows in {args.inputs}')

    fragment = build_fragment(rows, args.case, args.reference)
    print(json.dumps(fragment, indent=2))

    entry = next(iter(fragment.values()))
    pt = entry['_meta']['point']
    print(f'\n# op {next(iter(fragment))} @ '
          f'{pt["m"]}x{pt["k"]}x{pt["n"]} ({pt["algebra"]}), '
          f'vs {args.reference} (ratio <1 = nitrix faster):', file=sys.stderr)
    for dev in ('cpu', 'gpu'):
        b = entry.get(f'perf_{dev}_baseline')
        r = entry.get(f'perf_{dev}_ratio')
        if b is None and r is None:
            note = entry['_meta'].get('notes', {}).get(dev, 'no data')
            print(f'  {dev}: --  ({note})', file=sys.stderr)
        else:
            print(f'  {dev}: nitrix-jax = {r}x {b}', file=sys.stderr)

    if args.apply:
        if not args.out:
            raise SystemExit('--apply requires --out (never mutates in place)')
        merged = _apply(Path(args.apply), fragment)
        Path(args.out).write_text(json.dumps(merged, indent=2) + '\n')
        print(f'\nMerged copy written to {args.out} (review, then commit in '
              f'nitrix).', file=sys.stderr)


if __name__ == '__main__':
    main()
