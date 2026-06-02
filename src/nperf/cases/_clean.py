# -*- coding: utf-8 -*-
"""Shared nilearn ``signal.clean`` confound-regression floor.

``nilearn.signal.clean`` is the canonical fMRI nuisance-regression op. Both
``nitrix.linalg.residualise`` (regress out a confound design) and
``nitrix.signal.polynomial_detrend`` (== residualise against a Vandermonde
basis) are OLS confound regression, so ``clean(confounds=...)`` is the faithful
reference for each.

**Match-the-right-target (verified by probe).** ``clean`` bundles detrend +
standardize + confound-removal + filtering; to isolate the pure OLS residual
that nitrix computes we set ``detrend=False, standardize=False,
standardize_confounds=False, filter=False``. The last is load-bearing:
``standardize_confounds=True`` (the nilearn default) *demeans* the confound
columns, which changes the projection (our designs carry no constant column) --
it fails the oracle by ~1e3 rel_to_tol, whereas the un-standardized projection
matches to round-off. nilearn warns that confounds are un-standardized; that
warning is exactly the behaviour we want (a raw projection == nitrix), so it is
suppressed. nilearn computes in float64.

nilearn's own ``detrend=True`` is **linear only** (degree 1), so a degree-d
``polynomial_detrend`` is expressed the idiomatic nilearn way: confound
regression against the degree-d polynomial (Vandermonde) basis.

nilearn is imported lazily -- only the numpy worker runs this floor.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def nilearn_clean(signals_ft: Any, confounds_tk: Any) -> np.ndarray:
    '''OLS-residualise ``signals`` (features x time) against ``confounds``
    (time x k) via ``nilearn.signal.clean`` -- the pure projection (no
    detrend / standardize), so it matches nitrix's residual. Returns
    (features x time).'''
    import warnings

    from nilearn.signal import clean

    sig = np.asarray(signals_ft).T  # clean wants (time, features)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = clean(
            sig, confounds=np.asarray(confounds_tk),
            detrend=False, standardize=False, standardize_confounds=False,
            filter=False, ensure_finite=False,
        )
    return out.T
