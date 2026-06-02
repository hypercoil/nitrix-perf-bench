# -*- coding: utf-8 -*-
"""Run-provider registry (L2) — the cross-case half of the "baseline registry".

A baseline's *name* (`nitrix-jax`, `naive-dense`, `jnp-matmul`, …) is a
**case-local label**: it means "the JAX reference *of this op*", so it is only
meaningful next to a `case`, and every L4 row already carries `case` to
disambiguate it.  Registering bare baseline names globally would therefore
**collide** the moment a second case wants its own `nitrix-jax`.  So we do not.

What *is* genuinely cross-case — and what carries the framework + env-isolation
metadata the runner needs — is the **provider**: the way a baseline is run.
Providers are few and shared across all cases (`jax`, `numpy`; later `torch`,
`pyg`), so they live in a registry.  A `BuiltPoint` maps each case-local
baseline name to `(provider_id, run_fn)`; the runner resolves the provider here
to pick the sync hook / `jit` and (P2) the env to spawn in.

The **env-isolation marker** lives on the provider (it is an env property, not
a per-op one): `isolation='uv'` (default) vs `isolation='pixi'` (the DESIGN §7
escape hatch).  A `pixi` provider **must** carry a `pixi_reason` — the audit
trail that keeps pixi from becoming a silent second dispatch system.  All
current providers are `uv`; the pixi ones land with the P2 torch / PyG refs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Provider:
    id: str
    framework: str  # 'jax' | 'numpy' | 'torch' | 'cupy' -> core.SYNC hook
    isolation: str = 'uv'  # 'uv' | 'pixi' (env isolation; DESIGN §7)
    pixi_reason: Optional[str] = None  # required iff isolation == 'pixi'
    # resource class the provider *requires* to run (e.g. 'gpu' for a CUDA-only
    # reference like cupy, which -- unlike adaptive torch -- has no CPU path
    # and ignores JAX_PLATFORMS).  The runner skips a baseline on a platform
    # whose resource does not satisfy this (recorded platform_not_applicable),
    # so a GPU-only ref never runs on -- or mislabels itself as -- the CPU
    # platform.
    requires: Optional[str] = None  # None | 'cpu' | 'gpu'
    description: str = ''

    def __post_init__(self) -> None:
        if self.isolation not in ('uv', 'pixi'):
            raise ValueError(
                f"provider {self.id!r}: isolation must be 'uv' or 'pixi'."
            )
        if self.isolation == 'pixi' and not self.pixi_reason:
            raise ValueError(
                f"provider {self.id!r}: isolation='pixi' requires a "
                'pixi_reason (why PyPI/uv could not build it) -- an audit '
                'trail, not a silent second dispatcher (DESIGN §7).'
            )
        if self.requires not in (None, 'cpu', 'gpu'):
            raise ValueError(
                f"provider {self.id!r}: requires must be None, 'cpu' or 'gpu'."
            )


PROVIDERS: Dict[str, Provider] = {
    p.id: p
    for p in (
        Provider('jax', 'jax', description='in-tree JAX (uv group)'),
        Provider('numpy', 'numpy', description='host NumPy (uv)'),
        # SciPy references (ndimage / signal / sparse): host, numpy-array
        # semantics, no device sync -> the numpy framework + base env.  A
        # distinct id so the baseline's library is attributable.
        Provider('scipy', 'numpy', description='host SciPy (uv)'),
        # scikit-learn references (sklearn.metrics.pairwise -- the canonical
        # CPU kernels): same shape as the scipy provider (host, numpy-array
        # semantics, no device sync -> numpy framework + base env), a distinct
        # id so the kernel CPU floor is attributable to the standard library.
        Provider('sklearn', 'numpy', description='host scikit-learn (uv)'),
        # P2 cross-framework refs.  torch's *CPU* wheel is on the PyTorch
        # index and uv-installable, so it is a uv env -- a separate
        # interpreter (built by tools/setup_refs_env.sh; the runner selects
        # it via NPERF_PYTHON_TORCH), not a separate package manager.  It is
        # still isolated from nitrix's jax-only env (DESIGN §7): nitrix must
        # never import torch.
        Provider('torch', 'torch',
                 description='cross-framework ref; separate uv env '
                             '(tools/setup_refs_env.sh)'),
        # PyTorch Geometric: framework 'torch' (torch tensors + sync hook), so
        # it shares the torch refs env / interpreter.  Modern PyG (>=2.3) does
        # message passing on torch-native scatter_reduce, so core PyG installs
        # pure-Python via uv -- no compiled torch-scatter/torch-sparse, hence
        # *not* the pixi escape hatch (forcing pixi here would fabricate the
        # very need the pixi_reason guard exists to prevent).  pixi stays
        # reserved (DESIGN §7) for if a baseline ever needs those *compiled*
        # extensions (e.g. their fused CUDA segment-reduce for a GPU parity
        # run) and no portable PyPI wheel exists for the torch/CUDA pin.
        Provider('pyg', 'torch',
                 description='PyTorch Geometric ref; torch refs env '
                             '(tools/setup_refs_env.sh)'),
        # CuPy (Phase B): the GPU twin of numpy/scipy -- the apples-to-apples
        # on-target reference for the audit ops (cupy.cov, cuSOLVER lstsq,
        # cupyx.scipy.ndimage).  GPU-only (requires='gpu'); its own refs-cupy
        # env (DESIGN §7), selected per attempt via NPERF_PYTHON_CUPY.
        Provider('cupy', 'cupy', requires='gpu',
                 description='GPU reference (cupy / cupyx.scipy.ndimage); '
                             'refs-cupy env (NPERF_PYTHON_CUPY)'),
    )
}


def provider(provider_id: str) -> Provider:
    '''Look up a registered provider; clear error on an unknown id.'''
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise KeyError(
            f'unregistered provider {provider_id!r}; known: '
            f'{sorted(PROVIDERS)} (register it in nperf/providers.py)'
        ) from None


def framework_of(provider_id: str) -> str:
    '''The framework a provider runs under (selects the sync hook / jit).'''
    return provider(provider_id).framework


def requires_of(provider_id: str) -> Optional[str]:
    '''The resource class a provider needs to run (``None`` | 'cpu' | 'gpu').

    Used by the runner to skip a baseline on a platform that can't satisfy it
    (a GPU-only ref on the CPU platform) -- recorded as
    ``platform_not_applicable``, never silently dropped.'''
    return provider(provider_id).requires
