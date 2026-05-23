# -*- coding: utf-8 -*-
"""L4 row schema (core/results.py): freeze + JSON round-trip.

The schema is the artefact P0b freezes; these guard the frozen invariants
(version, status serialisation, jsonl round-trip).
"""
import json

from nperf.core.results import (
    SCHEMA_VERSION,
    AttemptRecord,
    Status,
    read_jsonl,
    write_jsonl,
)


def test_schema_frozen_at_1():
    assert SCHEMA_VERSION == 1


def _ok_record() -> AttemptRecord:
    return AttemptRecord(
        run_id='r', case='semiring_matmul',
        param_point={'m': 512, 'k': 512, 'n': 512, 'algebra': 'log'},
        baseline='nitrix-jax', platform='jax-cpu', framework='jax',
        status=Status.OK,
        metrics={'steady_time': {'min': 1.0, 'unit': 's'}},
        fidelity={'status': 'pass', 'rel_to_tol': 0.0},
        ratio={'vs': 'nitrix-jax', 'metric': 'min', 'value': 1.0},
    )


def test_status_serialises_to_value():
    d = _ok_record().to_json()
    assert d['status'] == 'ok'  # enum -> string value, not "Status.OK"
    assert d['schema_version'] == 1
    json.dumps(d)  # must be JSON-serialisable end to end


def test_from_json_roundtrip():
    rec = _ok_record()
    back = AttemptRecord.from_json(rec.to_json())
    assert back.status is Status.OK  # string -> enum
    assert back.to_json() == rec.to_json()  # lossless


def test_jsonl_roundtrip(tmp_path):
    recs = [
        _ok_record(),
        AttemptRecord(
            run_id='r', case='semiring_matmul', param_point={'m': 1024},
            baseline='naive-dense', platform='jax-cpu', framework='jax',
            status=Status.OOM,
            failure_detail={'message': 'resource_exhausted'},
        ),
    ]
    path = write_jsonl(recs, tmp_path / 'out.jsonl')
    back = read_jsonl(path)
    assert len(back) == 2
    assert back[0]['status'] == 'ok'
    assert back[1]['status'] == 'oom'
    # A non-ok row carries no metrics but keeps its failure_detail.
    assert back[1]['metrics'] is None
    assert back[1]['failure_detail']['message'] == 'resource_exhausted'
