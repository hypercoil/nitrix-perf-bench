# -*- coding: utf-8 -*-
"""Per-framework synchronisation hooks (L0).

A timed call is only honest if we block on the result before stopping the
clock — JAX dispatch is async, so without ``block_until_ready`` we would time
dispatch, not compute.  Each framework registers its sync; the timer is given
the right one per baseline (DESIGN §L0).
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import jax


def jax_sync(out: Any) -> None:
    '''Block on every leaf of a JAX output (handles pytrees and arrays).'''
    jax.block_until_ready(out)


def numpy_sync(out: Any) -> None:
    '''NumPy results are already host-materialised; nothing to wait on.'''
    return None


def torch_sync(out: Any) -> None:
    '''Block on a torch result before the clock stops.

    torch CUDA dispatch is async like JAX, so a GPU timing is only honest
    after ``torch.cuda.synchronize()``.  On CPU there is nothing to wait on
    (eager ops are synchronous), and ``synchronize`` is a no-op without CUDA,
    so this is correct on both.  Imported lazily so the (jax-only) base env
    never needs torch to load this module.
    '''
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cupy_sync(out: Any) -> None:
    '''Block on a CuPy result before the clock stops.

    CuPy kernel launches are async on the default stream, so a GPU timing is
    only honest after a device synchronise.  Imported lazily so the (jax-only)
    base env never needs cupy to load this module; cupy lives in its own
    refs-cupy env (DESIGN §7 / COVERAGE_MANDATE Thrust 3).
    '''
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


# torch (above) lands with the P2 cross-framework refs; ``pyg`` reuses it.
# cupy (Phase B) is the GPU reference for the audit ops (requires a GPU).
SYNC: Dict[str, Callable[[Any], None]] = {
    'jax': jax_sync,
    'numpy': numpy_sync,
    'torch': torch_sync,
    'cupy': cupy_sync,
}
