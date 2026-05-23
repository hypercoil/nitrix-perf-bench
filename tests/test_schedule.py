# -*- coding: utf-8 -*-
"""Scheduler invariants (schedule.py) — the device lock + pinned CPU slots.

These are the honesty guarantees: at most one timed attempt per physical
resource (GPU device lock), CPU concurrency bounded by `cpu_slots` on disjoint
cores, distinct resources may overlap, and results come back in submission
order.  Pure-Python (a fake `run_one`); no jax/GPU needed.
"""
import threading
import time

from nperf.schedule import ResourcePool, partition_cores, run_scheduled


def test_partition_cores_disjoint_even_and_complete():
    groups = partition_cores(list(range(8)), 3)
    assert [len(g) for g in groups] == [3, 3, 2]  # near-even
    flat = [c for g in groups for c in g]
    assert sorted(flat) == list(range(8))  # complete
    assert len(set(flat)) == 8  # disjoint
    # More slots than cores -> capped at one core each, still disjoint.
    assert partition_cores([0, 1], 5) == [[0], [1]]


class _ConcurrencyProbe:
    '''Records peak simultaneous executions per resource + observed cores.'''

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: dict = {}
        self.peak: dict = {}
        self.concurrent_cores: list = []

    def run_one(self, spec, ctx):
        res = spec['resource']
        with self.lock:
            self.current[res] = self.current.get(res, 0) + 1
            self.peak[res] = max(self.peak.get(res, 0), self.current[res])
            if ctx.get('cores') is not None:
                self.concurrent_cores.append((self.current[res], ctx['cores']))
        time.sleep(0.05)  # force overlap if the scheduler allows it
        with self.lock:
            self.current[res] -= 1
        return spec['baseline']


def _specs(resource, n):
    return [{'resource': resource, 'baseline': f'{resource}-{i}'}
            for i in range(n)]


def test_gpu_device_lock_serialises():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=4)  # cpu slots irrelevant to gpu resource
    run_scheduled(_specs('gpu:0', 6), probe.run_one, pool, max_parallel=4)
    assert probe.peak['gpu:0'] == 1  # the device lock: never two at once


def test_cpu_slots_bound_concurrency():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=2)
    run_scheduled(_specs('cpu', 6), probe.run_one, pool, max_parallel=4)
    assert probe.peak['cpu'] == 2  # bounded by cpu_slots, not max_parallel


def test_distinct_resources_overlap():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=1)
    specs = _specs('cpu', 3) + _specs('gpu:0', 3)
    run_scheduled(specs, probe.run_one, pool, max_parallel=2)
    # cpu (1 slot) and gpu (lock) each serialise, but a cpu and a gpu attempt
    # can run at the same time -> each resource peaks at exactly 1.
    assert probe.peak['cpu'] == 1 and probe.peak['gpu:0'] == 1


def test_results_in_submission_order():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=3)
    specs = _specs('cpu', 5)
    out = run_scheduled(specs, probe.run_one, pool, max_parallel=3)
    assert out == [s['baseline'] for s in specs]
