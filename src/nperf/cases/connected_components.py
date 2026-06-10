# -*- coding: utf-8 -*-
"""Morphology: ``nitrix.morphology.connected_components`` vs scipy / cupyx.

Label the connected foreground regions of a binary mask -- jit-able label
propagation with pointer jumping (``lax.while_loop``; O(log d) passes for a
diameter-d component), the interesting GPU-scaling op.

**Task-level (no elementwise oracle).**  The label *IDs* are
implementation-dependent (scipy, nitrix and cupyx each number the regions
differently -- a relabelling/permutation), so there is no bit-comparable
ground truth (``fp64_reference`` is ``None``).  What is invariant is the
**partition** -- which voxels are grouped together -- and that matches scipy
(verified, pinned in ``tests/test_connectivity_cases.py``).  scipy ``label``
(CPU floor) and cupyx ``label`` (GPU bar) ride as honest *perf* references
computing the same partition.  Ratio vs ``nitrix-jax``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from nitrix.morphology import connected_components

from ._base import BuiltPoint, Case, to_cupy
from ._connectivity import blob_mask, cupyx_label, scipy_label


def _build(param: Dict[str, Any]) -> BuiltPoint:
    d = param['d']
    mask = blob_mask((d, d, d), param.get('seed', 0))
    mj = jax.block_until_ready(jnp.asarray(mask))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(mask)
        return (mask,) if framework == 'numpy' else (mj,)

    baselines = {
        'nitrix-jax': ('jax', lambda m: connected_components(m)),
        'scipy.label': ('scipy', scipy_label),  # perf ref (same partition)
        'cupy.label': ('cupy', cupyx_label()),  # GPU perf ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('task-level: label IDs are implementation-dependent (a '
                       'permutation), so no elementwise oracle. The partition '
                       'matches scipy (pinned in tests); scipy/cupyx label '
                       'are perf references computing the same partition.'),
        ratio_reference='nitrix-jax',
    )


_SIZES = [32, 48, 64]
_LARGE = [96, 128, 160]

CASE = Case(
    name='connected_components',
    op_qualname='nitrix.morphology.connected_components',
    output_independent=False,  # global label propagation
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'d': d, 'seed': 0} for d in _SIZES],
    representative={'d': 48, 'seed': 0},
    large_param_points=tuple({'d': d, 'seed': 0} for d in _LARGE),
    complexity=(
        'jit-able label propagation with POINTER JUMPING (lax.while_loop: a '
        'neighbour-max hop + an L=L[L-1] pointer-jump per pass), O(log d) '
        'passes for diameter d, each O(N). MEASURED (L4): nitrix steady GROWS '
        'STEEPLY (1.3 -> 15.7 ms over 48 -> 160^3, ~12x) while cupyx label '
        'stays ~flat (0.6 -> 0.85 ms), so cupyx pulls from ~2x to ~18x ahead '
        '-- nitrix SCALES POORLY here, a kernel/algorithm scale risk (filed '
        'on nitrix main).'),
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
