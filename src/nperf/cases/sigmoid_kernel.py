# -*- coding: utf-8 -*-
"""Tier-2 kernel: ``nitrix.linalg.sigmoid_kernel`` vs sklearn / cupy.

The sigmoid (hyperbolic-tangent) kernel ``tanh(γ⟨x,y⟩ + r)`` -- nitrix vs
``sklearn.metrics.pairwise.sigmoid_kernel`` (r->coef0; verified exact in fp64)
+ a CuPy GPU ref. GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict

from nitrix.linalg import sigmoid_kernel

from ._base import BuiltPoint
from ._kernels import build_kernel_point, kernel_case

_GAMMA, _R = 0.1, 0.5


def _build(param: Dict[str, Any]) -> BuiltPoint:
    return build_kernel_point(
        lambda x: sigmoid_kernel(x, gamma=_GAMMA, r=_R),
        'sigmoid', param, gamma=_GAMMA, r=_R)


CASE = kernel_case('sigmoid_kernel', _build)
