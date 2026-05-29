# -*- coding: utf-8 -*-
"""Emit the coverage-&-deficit report (report/coverage.py; MANDATE §2.2).

Joins the nitrix op catalogue (``op_matrix.json``) with the perf-bench L4 store
and writes the ranked deficit report -- (a) under-covered ops by priority,
(b) measured-but-lagging ops by severity -- as markdown + JSON for the nitrix
agent.  It is the inverse of the op_matrix feed: the feed pushes the numbers we
have; this surfaces the gaps and the on-target deficits.

No measurement / GPU needed -- it reads the accumulated rows::

    JAX_PLATFORMS=cpu python tools/coverage_report.py
    JAX_PLATFORMS=cpu python tools/coverage_report.py --from reports/*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf import store  # noqa: E402
from nperf.core import read_jsonl  # noqa: E402
from nperf.measure import CASES  # noqa: E402
from nperf.report import coverage as cov  # noqa: E402

# The op catalogue lives in nitrix (capability-only); default to the sibling
# checkout.  We read it purely as the list of ops -- capability stays nitrix's.
_DEFAULT_OP_MATRIX = str(
    Path(__file__).resolve().parents[2] / 'nitrix' / 'docs' / 'op_matrix.json'
)


def _op_to_case() -> Dict[str, Tuple[str, Dict[str, Any]]]:
    '''op qualname -> (case name, representative point); the single home for
    the case->op mapping is ``Case.op_qualname`` (shared with the feed).'''
    return {
        c.op_qualname: (c.name, c.representative)
        for c in CASES.values() if c.op_qualname
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--op-matrix', default=_DEFAULT_OP_MATRIX,
                    help="nitrix's op_matrix.json (the op catalogue)")
    ap.add_argument('--from', dest='inputs', nargs='+',
                    default=[store.STORE_DEFAULT], metavar='PATH',
                    help='L4 rows: .jsonl files or store dirs (default: the '
                         'store). Newest row per key is used.')
    ap.add_argument('--out-md', default='reports/COVERAGE_DEFICIT.md')
    ap.add_argument('--out-json', default='reports/coverage_deficit.json')
    args = ap.parse_args()

    catalogue = json.loads(Path(args.op_matrix).read_text()).get('ops', [])
    rows: list = []
    for f in store.expand_inputs(args.inputs):
        rows.extend(read_jsonl(f))
    rows = store.latest(rows)
    records = cov.build_coverage(rows, catalogue, _op_to_case())

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(cov.render_markdown(records))
    doc = cov.render_json(records)
    Path(args.out_json).write_text(json.dumps(doc, indent=2) + '\n')

    s = doc['summary']
    print(f"coverage: {s['multiplatform']}/{s['runtime_ops']} multiplatform, "
          f"{s['with_strong_gpu_ref']} with a strong GPU ref, "
          f"{s['lagging_on_gpu']} lagging on GPU. "
          f"Wrote {args.out_md} + {args.out_json}.", file=sys.stderr)


if __name__ == '__main__':
    main()
