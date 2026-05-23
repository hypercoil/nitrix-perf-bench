# -*- coding: utf-8 -*-
"""The L4 result row (SCHEMA_AND_LIFECYCLE §A) — the source of truth.

One record per **measurement attempt** ``(case, param_point, baseline,
platform)``, with per-metric distributions nested under the attempt and a
``status`` enum.  A ``fidelity_failed`` / ``fidelity_inconclusive`` attempt
still carries its ``metrics`` (the absolutes were measured; only the *ratio*
was refused) — that is the property that keeps refusal from dropping data.

``SCHEMA_VERSION = 1`` is the **P0b frozen** schema: changes from here are
**additive-only** (new optional fields / new ``Status`` members), never a
rename or a removal, so older result rows stay readable.  (P0a shipped ``0``,
the *disposable* schema iterated freely against the throwaway case; the rows it
produced are not meant to be read back.)  The fidelity headline ``rel_to_tol``
— worst error as a multiple of the allowed tolerance, gate-consistent
(``pass`` ⟺ ``<= 1``) — was folded in during P0a validation and is part of the
frozen ``fidelity`` block (see ``fidelity.compare`` and
``SCHEMA_AND_LIFECYCLE.md`` §A/§C).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA_VERSION = 1  # P0b: FROZEN. Additive-only from here (was 0, disposable).


class Status(str, Enum):
    '''The outcome of a measurement attempt; selects which optional blocks of
    the row are present (see ``failure_detail`` shapes in the annex §A).'''

    OK = 'ok'
    ENV_FAILED = 'env_failed'
    COMPILE_ERROR = 'compile_error'
    OOM = 'oom'
    TIMEOUT = 'timeout'
    FIDELITY_FAILED = 'fidelity_failed'
    FIDELITY_INCONCLUSIVE = 'fidelity_inconclusive'
    SKIPPED = 'skipped'


@dataclass
class AttemptRecord:
    # ---- identity / keys ----
    run_id: str
    case: str
    param_point: Dict[str, Any]
    baseline: str
    platform: str
    framework: str
    # ---- outcome ----
    status: Status
    # present iff status in {ok, fidelity_failed, fidelity_inconclusive}
    metrics: Optional[Dict[str, Any]] = None
    # present iff a comparison was attempted
    fidelity: Optional[Dict[str, Any]] = None
    # present iff status == ok AND fidelity passed (computed in L1, stored)
    ratio: Optional[Dict[str, Any]] = None
    # ---- present iff status != ok; shape depends on status ----
    failure_detail: Optional[Dict[str, Any]] = None
    # ---- provenance (§1.1) ----
    provenance: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


def write_jsonl(records: Iterable[AttemptRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json()) + '\n')
    return path


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]
