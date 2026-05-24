# -*- coding: utf-8 -*-
"""Emit a decision-input bundle for one op at one decision point (P3).

A bundle is the *structured evidence* a human needs for a "benchmark-first"
call — the competing baselines' ratios, each one's fidelity + threshold check,
and the per-run trend — with **no recommendation** (DESIGN §1/§5).  It reads
accumulated L4 rows (the store, or explicit files) and renders JSON + markdown.

Run::

    JAX_PLATFORMS=cpu python tools/decision_bundle.py \
        --case semiring_matmul --from results/store/semiring_matmul
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import bundle, store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402
from nperf.report import render_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--case', default='semiring_matmul', choices=sorted(CASES))
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='L4 rows: .jsonl files or store dirs (default: the '
                         'store). Newest row per key = current; all runs = '
                         'the trend.')
    ap.add_argument('--point', default=None, metavar='JSON',
                    help='decision point (JSON); default: the case '
                         'representative')
    ap.add_argument('--bundle-out', default=None, metavar='PATH',
                    help='machine-readable bundle (default '
                         'results/<case>_bundle.json)')
    ap.add_argument('--report', default=None, metavar='PATH',
                    help='markdown bundle (default results/<case>_bundle.md)')
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for f in store.expand_inputs(args.inputs):
        rows.extend(read_jsonl(f))
    if not rows:
        raise SystemExit(f'no rows in {args.inputs}')
    point = json.loads(args.point) if args.point else None

    b = bundle.build_bundle(rows, case=args.case, point=point)
    out = Path(args.bundle_out or f'results/{args.case}_bundle.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(b, indent=2) + '\n')
    md = render_bundle(b)
    report = Path(args.report or f'results/{args.case}_bundle.md')
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(md)
    print(md)
    print(f'Wrote bundle {out} and {report}.', file=sys.stderr)


if __name__ == '__main__':
    main()
