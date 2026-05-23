# -*- coding: utf-8 -*-
"""Scheduler invariants (schedule.py) — the device lock + pinned CPU slots.

These are the honesty guarantees: at most one timed attempt per physical
resource (GPU device lock), CPU concurrency bounded by `cpu_slots` on disjoint
cores, distinct resources may overlap, and results come back in submission
order.  Pure-Python (a fake `run_one`); no jax/GPU needed.
"""
import threading
import time
from collections import Counter

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
    '''Records peak simultaneous executions per resource + devices in use.'''

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current: dict = {}
        self.peak: dict = {}
        self.held: Counter = Counter()  # device id -> live holders
        self.max_per_device = 0
        self.devices_seen: set = set()

    def run_one(self, spec, ctx):
        res = spec['resource']
        dev = ctx.get('device')
        with self.lock:
            self.current[res] = self.current.get(res, 0) + 1
            self.peak[res] = max(self.peak.get(res, 0), self.current[res])
            if dev is not None:
                self.held[dev] += 1
                self.max_per_device = max(self.max_per_device, self.held[dev])
                self.devices_seen.add(dev)
        time.sleep(0.05)  # force overlap if the scheduler allows it
        with self.lock:
            self.current[res] -= 1
            if dev is not None:
                self.held[dev] -= 1
        return spec['baseline']


def _specs(resource, n):
    return [{'resource': resource, 'baseline': f'{resource}-{i}'}
            for i in range(n)]


def test_gpu_single_device_lock_serialises():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(n_gpus=1)  # one device -> its lock serialises
    run_scheduled(_specs('gpu', 6), probe.run_one, pool, max_parallel=4)
    assert probe.peak['gpu'] == 1  # never two attempts on the one device


def test_multi_gpu_fans_out_one_attempt_per_device():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(n_gpus=3)
    run_scheduled(_specs('gpu', 9), probe.run_one, pool, max_parallel=8)
    assert probe.peak['gpu'] == 3  # all three devices busy at once
    assert probe.max_per_device == 1  # no device double-booked (per-dev lock)
    assert probe.devices_seen == {0, 1, 2}  # exactly the physical ids 0..2


def test_cpu_slots_bound_concurrency():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=2)
    run_scheduled(_specs('cpu', 6), probe.run_one, pool, max_parallel=4)
    assert probe.peak['cpu'] == 2  # bounded by cpu_slots, not max_parallel


def test_distinct_resources_overlap():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=1, n_gpus=1)
    specs = _specs('cpu', 3) + _specs('gpu', 3)
    run_scheduled(specs, probe.run_one, pool, max_parallel=2)
    # cpu (1 slot) and gpu (1 device) each serialise, but a cpu and a gpu
    # attempt can run at the same time -> each resource peaks at exactly 1.
    assert probe.peak['cpu'] == 1 and probe.peak['gpu'] == 1


def test_results_in_submission_order():
    probe = _ConcurrencyProbe()
    pool = ResourcePool(cpu_slots=3)
    specs = _specs('cpu', 5)
    out = run_scheduled(specs, probe.run_one, pool, max_parallel=3)
    assert out == [s['baseline'] for s in specs]
