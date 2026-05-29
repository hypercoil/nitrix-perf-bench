# -*- coding: utf-8 -*-
"""Fidelity comparison against the ground-truth oracle (L1 / L2).

Returns the *structured* record from SCHEMA_AND_LIFECYCLE §A/§C, never a bare
scalar — a single number hides exactly the disagreements we care about.  The
caller supplies the oracle output (computed once per param point, in fp64,
outside the timed region) and the per-case tolerance; this module only
compares already-layout-normalised arrays.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compare(
    out: Any,
    ref: Any,
    *,
    rtol: float,
    atol: float,
    oracle_kind: str = 'fp64_full',
    oracle_baseline: Optional[str] = None,
    oracle_subsample: Optional[Dict[str, Any]] = None,
    layout_normalised: bool = True,
) -> Dict[str, Any]:
    '''Compare a baseline output to the oracle; return the fidelity record.

    ``status`` is ``pass`` / ``fail`` (np.allclose semantics, per element) —
    ``inconclusive`` is decided upstream (e.g. a too-small subsample), not
    here.  The adapter is responsible for layout / index normalisation before
    calling this; we only verify shapes match.
    '''
    # fp64, or complex128 for complex outputs (e.g. analytic_signal): the error
    # magnitudes below (np.abs) are real either way, so rel_to_tol / max_abs
    # stay well-defined -- a complex output is compared by |out - ref|.
    hp = (np.complex128 if np.iscomplexobj(out) or np.iscomplexobj(ref)
          else np.float64)
    out_a = np.asarray(out, dtype=hp)
    ref_a = np.asarray(ref, dtype=hp)
    if out_a.shape != ref_a.shape:
        raise ValueError(
            f'fidelity.compare: shape mismatch {out_a.shape} vs {ref_a.shape}'
            ' -- the adapter must normalise layout before comparing.'
        )
    diff = np.abs(out_a - ref_a)
    denom = np.abs(ref_a)
    tol = atol + rtol * denom
    within = diff <= tol
    n_mismatched = int((~within).sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    # Headline: worst error as a multiple of the *allowed* tolerance.
    # Well-defined everywhere (tol >= atol > 0) and gate-consistent:
    # pass <=> rel_to_tol <= 1.  This is the scalar to read; a plain
    # element-wise max_rel is meaningless for zero-centred outputs (matmul,
    # residuals, normalised data) where most |ref| are near zero.
    rel_to_tol = float(np.max(diff / tol)) if diff.size else 0.0
    # max_rel kept for completeness, guarded to significant ref elements.
    sig = denom > atol
    max_rel = float(np.max(diff[sig] / denom[sig])) if np.any(sig) else 0.0
    return {
        'status': 'pass' if n_mismatched == 0 else 'fail',
        'rel_to_tol': rel_to_tol,
        'max_abs': max_abs,
        'max_rel': max_rel,
        'n_mismatched': n_mismatched,
        'layout_normalised': layout_normalised,
        'oracle': {
            'kind': oracle_kind,
            'baseline': oracle_baseline,
            'subsample': oracle_subsample,
        },
        'threshold': {'rtol': rtol, 'atol': atol, 'scope': 'per_case'},
    }
