# -*- coding: utf-8 -*-
"""Resource-aware scheduler + per-physical-device lock (L3, annex §E).

A **physical resource** is a contention domain that must host **at most one
timed attempt at a time** for the measurement to be honest.  This module owns
that invariant, *above* the worker-spawn path (dispatcher-agnostic: it does not
care whether a worker is a uv interpreter, a prebuilt env, or pixi):

- **GPU** → one permit **per physical device** (`gpu:0 … gpu:N-1`).  Each
  permit *is* that device's lock: a device hosts one attempt at a time
  (back-to-back, clock state stable, optional settle), but **N devices run N
  attempts concurrently**.  A `gpu` attempt is handed *any* free device id; the
  worker is pinned to it via `CUDA_VISIBLE_DEVICES`.
- **CPU** → ``cpu_slots`` permits, each bound to a **disjoint core group**.
  Parallel CPU attempts then run on non-overlapping cores (pinned in the
  worker, `worker.py`), so concurrency is free of cross-attempt *contention*.
  Note the timings then reflect each slot's **core budget** (recorded), not the
  whole machine; ``cpu_slots = 1`` (default) is plain full-machine serial
  timing.

A `resource` is now a **class** (`'cpu'` / `'gpu'`); the pool hands out a
specific permit (a core group or a device id).  Attempts on **distinct**
resources — and on **distinct GPUs** — proceed concurrently; the scheduler is a
thread pool whose tasks each acquire a permit around the (blocking) run.
"""
from __future__ import annotations

import os
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
    '''The physical resource *class* a platform's attempts contend for.'''
    return 'cpu' if platform == 'jax-cpu' else 'gpu'


class ResourcePool:
    '''Per-resource permits: per-device GPU locks + pinned CPU slots.'''

    def __init__(
        self, *, cpu_slots: int = 1, n_gpus: int = 1,
        gpu_settle_s: float = 0.0,
    ) -> None:
        cores = _available_cores()
        self.cpu_slots = max(1, min(cpu_slots, len(cores)))
        self.n_gpus = max(0, n_gpus)
        self.gpu_settle_s = gpu_settle_s
        # Each CPU permit carries a disjoint core group (the pinning target);
        # each GPU permit is a physical device id (its presence = its lock).
        self.core_groups = partition_cores(cores, self.cpu_slots)
        self._cpu_q: "Queue[List[int]]" = Queue()
        for group in self.core_groups:
            self._cpu_q.put(group)
        self._gpu_q: "Queue[int]" = Queue()
        for dev in range(self.n_gpus):
            self._gpu_q.put(dev)

    def permits(self, resource: str) -> int:
        '''How many attempts may run at once on this resource class.'''
        return self.cpu_slots if resource == 'cpu' else self.n_gpus

    @contextmanager
    def acquire(self, resource: str) -> Iterator[Dict[str, Any]]:
        '''Hold one permit of the resource class; yield its execution context.

        CPU yields the pinned ``cores``; GPU yields a ``device`` id (the worker
        pins to it via ``CUDA_VISIBLE_DEVICES``), with the settle interval held
        *inside* the per-device lock so the next attempt starts on a rested
        device.  Blocks until a permit is free (≤ one attempt per device).'''
        if resource == 'cpu':
            cores = self._cpu_q.get()
            try:
                yield {'cores': cores, 'device': None}
            finally:
                self._cpu_q.put(cores)
        else:
            dev = self._gpu_q.get()  # the per-device lock (annex §E)
            try:
                yield {'cores': None, 'device': dev}
            finally:
                if self.gpu_settle_s:
                    time.sleep(self.gpu_settle_s)
                self._gpu_q.put(dev)


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
