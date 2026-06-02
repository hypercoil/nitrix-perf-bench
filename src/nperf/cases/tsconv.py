# -*- coding: utf-8 -*-
"""Tier-2 (B11 breadth): ``nitrix.signal.tsconv`` vs scipy / cupy.

1-D convolution along the trailing axis (SAME padding).  Empirically tsconv is
a **cross-correlation** (conv-layer / torch.conv1d convention -- the kernel is
*not* flipped) with the kernel centred on each output sample: nitrix (jax) vs
``scipy.signal.correlate(mode='same')`` (the CPU floor) +
``cupyx.scipy.signal.correlate`` (GPU ref), scored against an fp64 oracle.
Convolution is GPU-pure (no cuSolver): a clean apples-to-apples GPU bar.

Scope: this case is **single-channel** (C_in = C_out = 1) so the reference is a
single vectorised ``correlate`` call -- a fair comparison free of Python
per-channel-pair loop overhead.  The genuinely multi-channel form
(``weight`` (C_out, C_in, K)) is a conv *layer*; its fair reference is
``torch.nn.functional.conv1d`` (a refs-env follow-up), not a scipy loop.  Ratio
vs ``scipy.signal.correlate``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as ss
from nitrix.signal import tsconv

from ._base import BuiltPoint, Case, to_cupy


def _scipy_corr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    '''Single-channel SAME cross-correlation, shaped back to (1, obs).'''
    return ss.correlate(x[0], w[0, 0], mode='same')[None, :]


def _cupy_corr(x: Any, w: Any) -> Any:
    '''GPU twin of ``_scipy_corr`` (cupy lazy; refs-cupy env).'''
    from cupyx.scipy.signal import correlate

    return correlate(x[0], w[0, 0], mode='same')[None, :]


def _build(param: Dict[str, Any]) -> BuiltPoint:
    obs, k = param['obs'], param['k']
    rng = np.random.default_rng(param.get('seed', 0))
    X = rng.standard_normal((1, obs)).astype(np.float32)
    W = rng.standard_normal((1, 1, k)).astype(np.float32)
    jx = jax.block_until_ready(jnp.asarray(X))
    jw = jax.block_until_ready(jnp.asarray(W))

    # fp64 oracle
    ref = _scipy_corr(X.astype(np.float64), W.astype(np.float64))

    def inputs_for(framework: str) -> Tuple[Any, ...]:
        if framework == 'cupy':
            return to_cupy(X, W)
        return (X, W) if framework == 'numpy' else (jx, jw)

    baselines = {
        'nitrix-jax': ('jax', lambda x, w: tsconv(x, w, padding='SAME')),
        'scipy.signal.correlate': ('scipy', _scipy_corr),
        'cupyx.scipy.signal.correlate': ('cupy', _cupy_corr),  # GPU ref
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for,
        fp64_reference=ref, ratio_reference='nitrix-jax',
    )


# (observations, kernel): cost ~ obs * K (a direct correlation per sample).
_SHAPES = [(4096, 15), (16384, 63), (65536, 127)]

CASE = Case(
    name='tsconv',
    op_qualname='nitrix.signal.tsconv',
    output_independent=False,  # each output is a K-window over the input
    metrics=['steady_time', 'compile_time', 'peak_hbm', 'host_rss',
             'throughput'],
    param_points=[{'obs': o, 'k': k, 'seed': 0} for (o, k) in _SHAPES],
    representative={'obs': 16384, 'k': 63, 'seed': 0},
    build=_build,
    rtol=1e-3,
    atol=1e-4,
)
