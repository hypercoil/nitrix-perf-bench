# -*- coding: utf-8 -*-
"""The rigorous stopwatch (L0).

First call is timed separately as ``compile_time`` (trace + compile +
execute); the remaining warm-up runs are discarded; the post-warm-up runs are
the steady-state distribution.  The clock always stops *after* the
framework-specific ``sync`` so async dispatch is not what we measure
(DESIGN §L0).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Tuple

from .stats import Distribution
from .sync import jax_sync


def bench_call(
    fn: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    *,
    warmup: int = 3,
    repeats: int = 10,
    sync: Callable[[Any], None] = jax_sync,
) -> Tuple[float, Distribution]:
    '''Time ``fn(*args)`` with warm-up.

    Returns ``(compile_s, steady)`` where ``compile_s`` is the first-call
    wall-time (compile + execute) and ``steady`` is the distribution of the
    post-warm-up timed runs.  ``fn`` is expected to be already ``jax.jit``-
    wrapped (jax baselines) so the timed loop hits the compile cache rather
    than re-tracing.

    Parameters
    ----------
    fn
        The callable under test.
    args
        Pre-built, on-device arguments (excludes host->device transfer from
        the timed region).
    warmup
        Un-timed warm-up runs; must be >= 1 so the first-call compile is
        excluded from ``steady``.
    repeats
        Timed runs after warm-up.
    sync
        Framework sync hook applied to the output before the clock stops.
    '''
    if warmup < 1:
        raise ValueError('warmup must be >= 1 to exclude compile time')
    t0 = time.perf_counter()
    out = fn(*args)
    sync(out)
    compile_s = time.perf_counter() - t0
    for _ in range(warmup - 1):
        out = fn(*args)
        sync(out)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        sync(out)
        samples.append(time.perf_counter() - t0)
    return compile_s, Distribution.from_samples(samples)
