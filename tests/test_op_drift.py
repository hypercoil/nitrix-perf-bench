# -*- coding: utf-8 -*-
"""Fast drift gate: nitrix op signatures must match the committed manifest.

This is the cheap, non-flaky half of ``tools/drift_check.py`` (signature
fingerprints; no case builds, no op execution), run in the default test suite
so an API / default-value change in nitrix that silently invalidates a case --
the class that broke bilateral (``sigma_features`` -> ``metric``) and flipped
``sosfilt``'s ``backend`` default to ``'auto'`` -- fails loudly here.

The *behavioural* half (output digests on the representative point) is heavier
and is the sprint / pre-bench gate: run ``JAX_PLATFORMS=cpu python
tools/drift_check.py``.  After reviewing flagged cases, re-bless both with
``tools/drift_check.py --update``.

Exemptions: ops under active nitrix refactor (the spectral eigensolver is being
rehomed to ``linalg`` with a dispatcher cleanup) churn by design; they are
excluded here and re-blessed once the refactor settles.  See the perf-bench
hardening notes.
"""
import json
import sys
from pathlib import Path

import pytest

from nperf.measure import CASES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import drift_check  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / 'reports' / \
    'op_drift_manifest.json'

# Ops under active nitrix refactor -- expected to drift; re-bless when settled.
# The spectral eigensolver rehome to ``linalg._eigsolve`` landed 2026-06-07 and
# is drift-confirmed stable, so laplacian_eigenmap / diffusion_embedding are now
# back under the gate (B18 Win 4).  The geometry interpolation-dispatcher (a new
# resampling-strategy feature) is in flight as of 2026-06-07, churning the
# ``resample`` / ``spatial_transform`` signatures; exempt until it settles, then
# re-bless and remove (mirroring how the eigensolver entries were handled).
_REFACTOR_EXEMPT = {
    'nitrix.geometry.resample',
    'nitrix.geometry.spatial_transform',
}

# manifest 'ops' is keyed by case name (qualname is a field).
_MANIFEST_OPS = json.loads(MANIFEST.read_text()).get('ops', {})


def _checked_cases():
    return sorted(
        c.name for c in CASES.values()
        if c.op_qualname and c.op_qualname not in _REFACTOR_EXEMPT
    )


@pytest.mark.parametrize('case_name', _checked_cases())
def test_signature_matches_manifest(case_name):
    assert case_name in _MANIFEST_OPS, (
        f'{case_name} has no manifest entry -- seed it with '
        f'`python tools/drift_check.py --update`'
    )
    stored = _MANIFEST_OPS[case_name].get('signature')
    if stored is None:
        pytest.skip(f'{case_name}: no signature recorded')
    current = drift_check._signature(CASES[case_name].op_qualname)
    assert current == stored, (
        f'{case_name} signature drifted (nitrix changed under the case):\n'
        f'  manifest: {stored}\n  current : {current}\n'
        f'Review the case, then re-bless with '
        f'`python tools/drift_check.py --update`.'
    )
