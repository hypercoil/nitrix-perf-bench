# -*- coding: utf-8 -*-
"""Shared helpers for the EDT + connectivity family (mask-in ops).

``nitrix.morphology.{distance_transform_edt, connected_components,
largest_connected_component}`` operate on a binary mask ``(*spatial,)``:

- **distance_transform_edt** -- the *exact* Euclidean distance transform, which
  nitrix documents as matching ``scipy.ndimage.distance_transform_edt``, so
  scipy is a **co-oracle** (not just a floor) and cupyx is the on-target
  GPU bar -- a real domain-tool kernel-vs-kernel.  (Distinct from the benched
  semiring ``distance_transform``.)
- **largest_connected_component** -- a *boolean* mask (the single biggest
  region), so it is permutation-invariant: scipy ``label`` + argmax-size is a
  clean co-oracle, exact.
- **connected_components** -- *integer labels*, whose IDs are
  implementation-dependent (a relabelling/permutation): there is no elementwise
  oracle, so it is task-level (``fp64_reference=None``) -- the *partition*
  (which voxels group together) matches scipy and is pinned in the tests; scipy
  / cupyx ``label`` ride as perf references.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def blob_mask(spatial: Sequence[int], seed: int = 0, sigma: float = 1.5,
              quantile: float = 0.7) -> np.ndarray:
    '''A blobby binary mask -- smoothed noise thresholded at a quantile, giving
    several connected foreground components of varying size (one dominant, so
    ``largest_connected_component`` is unambiguous).'''
    import scipy.ndimage as ndi

    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(
        rng.standard_normal(tuple(spatial)).astype(np.float32), sigma)
    return f > np.quantile(f, quantile)


# ---- scipy CPU floors / fp64 oracles -------------------------------------- #
def scipy_edt(mask: Any) -> np.ndarray:
    import scipy.ndimage as ndi

    return ndi.distance_transform_edt(np.asarray(mask))


def scipy_largest_cc(mask: Any) -> np.ndarray:
    import scipy.ndimage as ndi

    lab, n = ndi.label(np.asarray(mask))
    if not n:
        return np.zeros_like(lab, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def scipy_label(mask: Any) -> np.ndarray:
    import scipy.ndimage as ndi

    return ndi.label(np.asarray(mask))[0]


# ---- cupyx GPU references ------------------------------------------------ #
def cupyx_edt() -> Callable[[Any], Any]:
    def run(mask: Any) -> Any:
        import cupyx.scipy.ndimage as c

        return c.distance_transform_edt(mask)

    return run


def cupyx_largest_cc() -> Callable[[Any], Any]:
    def run(mask: Any) -> Any:
        import cupy as cp
        import cupyx.scipy.ndimage as c

        lab, n = c.label(mask)
        sizes = cp.bincount(lab.ravel())
        sizes[0] = 0
        return lab == int(sizes.argmax())

    return run


def cupyx_label() -> Callable[[Any], Any]:
    def run(mask: Any) -> Any:
        import cupyx.scipy.ndimage as c

        return c.label(mask)[0]

    return run
