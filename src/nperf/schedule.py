# -*- coding: utf-8 -*-
"""Resource-aware scheduler + per-physical-device lock (L3, annex §E).

A **physical resource** is a contention domain that must host **at most one
timed attempt at a time** for the measurement to be honest.  This module owns
that invariant, *above* the worker-spawn path (dispatcher-agnostic: it does not
care whether a worker is a uv interpreter, a prebuilt env, or pixi):

- **GPU** → a 1-permit lock (the *device lock*).  Attempts on one device
  serialise back-to-back so they never corrupt each other's timings and the
  clock state stays stable; an optional settle interval is held between them.
- **CPU** → ``cpu_slots`` permits, each bound to a **disjoint core group**.
  Parallel CPU attempts then run on non-overlapping cores (pinned in the
  worker, `worker.py`), so concurrency is free of cross-attempt *contention*.
  Note the timings then reflect each slot's **core budget** (recorded), not the
  whole machine; ``cpu_slots = 1`` (default) is plain full-machine serial
  timing.

Attempts on **distinct** resources proceed concurrently; the scheduler is a
thread pool whose tasks each acquire their resource's permit around the
(blocking) subprocess run.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence


def _available_cores() -> List[int]:
    if hasattr(os, 'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def partition_cores(cores: Sequence[int], k: int) -> List[List[int]]:
    '''Split ``cores`` into ``k`` near-even **disjoint** contiguous groups.'''
    k = max(1, min(k, len(cores)))
    base, rem = divmod(len(cores), k)
    out: List[List[int]] = []
    i = 0
    for g in range(k):
        size = base + (1 if g < rem else 0)
        out.append(list(cores[i:i + size]))
        i += size
    return out


def resource_of(platform: str) -> str:
    '''The physical resource a platform's attempts contend for.'''
    return 'cpu' if platform == 'jax-cpu' else 'gpu:0'


class ResourcePool:
    '''Per-resource permits: the GPU device lock + pinned CPU slots.'''

    def __init__(
        self, *, cpu_slots: int = 1, gpu_settle_s: float = 0.0,
        gpus: Sequence[str] = ('gpu:0',),
    ) -> None:
        cores = _available_cores()
        self.cpu_slots = max(1, min(cpu_slots, len(cores)))
        self.gpu_settle_s = gpu_settle_s
        # Each CPU permit carries a disjoint core group (the pinning target).
        self._core_groups: "Queue[List[int]]" = Queue()
        self.core_groups = partition_cores(cores, self.cpu_slots)
        for group in self.core_groups:
            self._core_groups.put(group)
        self._gpu_locks = {g: threading.Semaphore(1) for g in gpus}

    def permits(self, resource: str) -> int:
        '''How many attempts may run at once on this resource.'''
        return self.cpu_slots if resource == 'cpu' else 1

    @contextmanager
    def acquire(self, resource: str) -> Iterator[Dict[str, Any]]:
        '''Hold the resource for one attempt; yield its execution context.

        For CPU the context carries the pinned ``cores``; for GPU it is the
        device lock (no cores; the worker uses the device), with the settle
        interval held *inside* the lock so the next attempt starts on a rested
        device.'''
        if resource == 'cpu':
            cores = self._core_groups.get()
            try:
                yield {'cores': cores}
            finally:
                self._core_groups.put(cores)
        else:
            lock = self._gpu_locks[resource]  # the device lock (annex §E)
            lock.acquire()
            try:
                yield {'cores': None}
            finally:
                if self.gpu_settle_s:
                    time.sleep(self.gpu_settle_s)
                lock.release()


def run_scheduled(
    specs: Sequence[Dict[str, Any]],
    run_one: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    pool: ResourcePool,
    *,
    max_parallel: int,
) -> List[Any]:
    '''Run ``run_one(spec, ctx)`` for every spec, honouring resource permits.

    Returns results in **submission order** (so the caller can slice them back
    to param points).  Each task blocks on its resource permit, so
    same-resource attempts serialise while distinct-resource ones overlap.
    '''
    results: List[Optional[Any]] = [None] * len(specs)

    def _task(index: int, spec: Dict[str, Any]) -> None:
        with pool.acquire(spec['resource']) as ctx:
            results[index] = run_one(spec, ctx)

    with ThreadPoolExecutor(max_workers=max(1, max_parallel)) as ex:
        futures = [ex.submit(_task, i, s) for i, s in enumerate(specs)]
        for fut in futures:
            fut.result()  # surface any unexpected scheduler-side exception
    return results
