# -*- coding: utf-8 -*-
"""Durable results store (store.py): accumulation, latest-per-key, retention.

A run is the unit of ingest (one file per run); separate runs/devices
accumulate; `latest` collapses to the current state per key; `prune` caps
history.  Pure filesystem + dicts — no jax.
"""
from nperf import store
from nperf.core import AttemptRecord, Status


def _rec(platform, baseline, run_id, case='c'):
    return AttemptRecord(
        run_id=run_id, case=case, param_point={'m': 8}, baseline=baseline,
        platform=platform, framework='jax', status=Status.OK,
    )


def test_ingest_and_load_roundtrip(tmp_path):
    p = store.ingest([_rec('jax-cpu', 'a', 'r1')],
                     root=str(tmp_path), case='c', run_id='r1')
    assert p == tmp_path / 'c' / 'r1.jsonl' and p.exists()
    rows = store.load(str(tmp_path), 'c')
    assert len(rows) == 1 and rows[0]['baseline'] == 'a'


def test_load_combines_separate_runs(tmp_path):
    store.ingest([_rec('jax-cpu', 'a', 'r1')],
                 root=str(tmp_path), case='c', run_id='r1')
    store.ingest([_rec('jax-cuda12', 'a', 'r2')],
                 root=str(tmp_path), case='c', run_id='r2')
    rows = store.load(str(tmp_path), 'c')
    assert len(rows) == 2
    assert {r['platform'] for r in rows} == {'jax-cpu', 'jax-cuda12'}


def test_latest_keeps_newest_per_key(tmp_path):
    old, new = '20260101T000000__x', '20260102T000000__y'
    for rid in (old, new):  # same (case, platform, param, baseline) twice
        store.ingest([_rec('jax-cpu', 'a', rid)],
                     root=str(tmp_path), case='c', run_id=rid)
    latest = store.load(str(tmp_path), 'c', latest_only=True)
    assert len(latest) == 1 and latest[0]['run_id'] == new  # newest wins
    assert len(store.load(str(tmp_path), 'c')) == 2  # full history kept


def test_prune_keeps_newest_n(tmp_path):
    for ts in ('20260101T000000__a', '20260102T000000__b',
               '20260103T000000__c'):
        store.ingest([_rec('jax-cpu', 'x', ts)],
                     root=str(tmp_path), case='c', run_id=ts)
    removed = store.prune(str(tmp_path), 'c', keep=2)
    assert len(removed) == 1
    kept = [f.stem for f in store.run_files(str(tmp_path), 'c')]
    assert kept == ['20260102T000000__b', '20260103T000000__c']  # oldest gone


def test_expand_inputs_globs_dir_and_passes_files(tmp_path):
    for rid in ('r1', 'r2'):
        store.ingest([_rec('jax-cpu', 'x', rid)],
                     root=str(tmp_path), case='c', run_id=rid)
    globbed = store.expand_inputs([str(tmp_path)])  # store root -> all runs
    assert len(globbed) == 2 and all(f.suffix == '.jsonl' for f in globbed)
    one = tmp_path / 'c' / 'r1.jsonl'
    assert store.expand_inputs([str(one)]) == [one]  # a file passes through
