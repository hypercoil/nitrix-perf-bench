# -*- coding: utf-8 -*-
"""Tier-2 kernel: ``nitrix.linalg.polynomial_kernel`` vs sklearn / cupy.

The polynomial kernel ``(γ⟨x,y⟩ + r)^order`` -- nitrix vs
``sklearn.metrics.pairwise.polynomial_kernel`` (order->degree, r->coef0;
verified exact in fp64) + a CuPy GPU ref. GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict

from nitrix.linalg import polynomial_kernel

from ._base import BuiltPoint
from ._kernels import build_kernel_point, kernel_case

_GAMMA, _ORDER, _R = 0.1, 3, 1.0


def _build(param: Dict[str, Any]) -> BuiltPoint:
    return build_kernel_point(
        lambda x: polynomial_kernel(x, gamma=_GAMMA, order=_ORDER, r=_R),
        'polynomial', param, gamma=_GAMMA, order=_ORDER, r=_R)


# atol a touch looser: the degree-order power amplifies the fp32 rounding of
# the inner products (still rel_to_tol << 1 -- the fp64 convention is exact).
CASE = kernel_case('polynomial_kernel', _build, atol=1e-3)
