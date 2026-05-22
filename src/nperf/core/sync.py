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


# Extended in P2 with ``torch`` (``torch.cuda.synchronize``), etc.
SYNC: Dict[str, Callable[[Any], None]] = {
    'jax': jax_sync,
    'numpy': numpy_sync,
}
