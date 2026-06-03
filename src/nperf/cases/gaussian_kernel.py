# -*- coding: utf-8 -*-
"""Tier-2 kernel: ``nitrix.linalg.gaussian_kernel`` vs sklearn / cupy.

The Gaussian (RBF) kernel ``exp(-dist^2 / 2*sigma^2)`` -- nitrix's sigma-
parameterised alias of ``rbf_kernel``. Reference:
``sklearn.metrics.pairwise.rbf_kernel`` with the matched ``gamma =
1/(2*sigma^2)`` (verified exact in fp64) + a CuPy GPU ref.
GPU-pure (matmul/broadcast); see ``cases/_kernels.py``. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict

from nitrix.linalg import gaussian_kernel

from ._base import BuiltPoint
from ._kernels import build_kernel_point, kernel_case

_SIGMA = 2.0


def _build(param: Dict[str, Any]) -> BuiltPoint:
    return build_kernel_point(
        lambda x: gaussian_kernel(x, sigma=_SIGMA), 'gaussian', param,
        gamma=1.0 / (2.0 * _SIGMA ** 2))


CASE = kernel_case('gaussian_kernel', _build)
