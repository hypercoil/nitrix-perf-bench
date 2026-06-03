# -*- coding: utf-8 -*-
"""Tier-2 kernel: ``nitrix.linalg.cosine_kernel`` vs sklearn / cupy.

Cosine similarity ``⟨x,y⟩ / (‖x‖‖y‖)`` (linear kernel on row-normalised rows).
Reference: ``sklearn.metrics.pairwise.cosine_similarity`` (CPU floor + fp64
oracle) + a CuPy GPU ref (Gram / outer norms). GPU-pure. Ratio vs nitrix-jax.
"""
from __future__ import annotations

from typing import Any, Dict

from nitrix.linalg import cosine_kernel

from ._base import BuiltPoint
from ._kernels import build_kernel_point, kernel_case


def _build(param: Dict[str, Any]) -> BuiltPoint:
    return build_kernel_point(lambda x: cosine_kernel(x), 'cosine', param)


CASE = kernel_case('cosine_kernel', _build)
