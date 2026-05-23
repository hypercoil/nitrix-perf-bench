# -*- coding: utf-8 -*-
"""Durable results store (L4 persistence).

A run's L4 rows are **append-only history**, one file per run:

    <root>/<case>/<run_id>.jsonl

`run_id` leads with the UTC timestamp (`YYYYMMDDThhmmss…__sha`), so lexical
sort == chronological and run files never collide.  A single run may already
span platforms (`--platforms`) and devices, so the *run* is the unit of ingest;
accumulating a second device's run (e.g. a Lovelace L40 beside the A10G) is
just another file — it never overwrites.

Reading back, `latest()` keeps the newest row per
`(case, platform, param_point, baseline)` so a combined report shows the
current state across devices; pass everything through for full history.
`prune()` caps history at the N most recent runs (the failure-row-volume guard,
DESIGN §8).

This module owns *on-disk accumulation + selection + retention*.  Where the
store lives and how it is shared across machines (local + git-ignored, a
results branch, or a network/object store) is the orthogonal transport policy
still open in DESIGN §8; the default root is the git-ignored ``results/store``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .core import read_jsonl, write_jsonl
from .core.results import AttemptRecord

STORE_DEFAULT = 'results/store'


def _key(row: Dict[str, Any]) -> tuple:
    '''Identity of a measurement across runs (newest wins in `latest`).'''
    return (
        row.get('case'), row.get('platform'),
        json.dumps(row.get('param_point'), sort_keys=True),
        row.get('baseline'),
    )


def ingest(
    records: Iterable[AttemptRecord], *, root: str, case: str, run_id: str
) -> Path:
    '''Append a run's rows to ``<root>/<case>/<run_id>.jsonl``.'''
    return write_jsonl(records, Path(root) / case / f'{run_id}.jsonl')


def run_files(root: str, case: str) -> List[Path]:
    '''Run files for a case, oldest-first (run_id is timestamp-leading).'''
    case_dir = Path(root) / case
    return sorted(case_dir.glob('*.jsonl')) if case_dir.is_dir() else []


def latest(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    '''Keep the newest row per `(case, platform, param_point, baseline)`.'''
    best: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        k = _key(r)
        if k not in best or r.get('run_id', '') > best[k].get('run_id', ''):
            best[k] = r
    return list(best.values())


def load(
    root: str, case: str, *, latest_only: bool = False
) -> List[Dict[str, Any]]:
    '''Read every run for a case; optionally collapse to newest-per-key.'''
    rows: List[Dict[str, Any]] = []
    for path in run_files(root, case):
        rows.extend(read_jsonl(path))
    return latest(rows) if latest_only else rows


def prune(root: str, case: str, keep: int) -> List[Path]:
    '''Delete all but the ``keep`` most recent runs; return the removed.'''
    files = run_files(root, case)
    victims = files[:-keep] if keep > 0 else []
    for f in victims:
        f.unlink()
    return victims


def expand_inputs(paths: Iterable[str]) -> List[Path]:
    '''Resolve render inputs: each path is a `.jsonl` file or a directory
    (globbed recursively for `*.jsonl`, e.g. a store root or a case dir).'''
    out: List[Path] = []
    for p in paths:
        path = Path(p)
        out.extend(sorted(path.rglob('*.jsonl')) if path.is_dir() else [path])
    return out
