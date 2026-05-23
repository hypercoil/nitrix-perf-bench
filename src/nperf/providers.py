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
    framework: str  # 'jax' | 'numpy' | 'torch' -> selects core.SYNC hook
    isolation: str = 'uv'  # 'uv' | 'pixi' (env isolation; DESIGN §7)
    pixi_reason: Optional[str] = None  # required iff isolation == 'pixi'
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


PROVIDERS: Dict[str, Provider] = {
    p.id: p
    for p in (
        Provider('jax', 'jax', description='in-tree JAX (uv group)'),
        Provider('numpy', 'numpy', description='host NumPy (uv)'),
        # P2 cross-framework refs.  torch's *CPU* wheel is on the PyTorch
        # index and uv-installable, so it is a uv env -- a separate
        # interpreter (built by tools/setup_refs_env.sh; the runner selects
        # it via NPERF_PYTHON_TORCH), not a separate package manager.  It is
        # still isolated from nitrix's jax-only env (DESIGN §7): nitrix must
        # never import torch.
        Provider('torch', 'torch',
                 description='cross-framework ref; separate uv env '
                             '(tools/setup_refs_env.sh)'),
        # PyG's compiled scatter/segment extensions (torch-scatter etc.) are
        # not on PyPI as portable wheels for an arbitrary torch/CUDA pin --
        # the conda-forge build is the supported path -- so it is the DESIGN
        # §7 pixi escape hatch, with this reason as its audit trail.
        Provider('pyg', 'torch', isolation='pixi',
                 pixi_reason='torch-scatter/torch-sparse compiled extensions '
                             'ship as conda-forge builds, not portable PyPI '
                             'wheels for an arbitrary torch/CUDA pin',
                 description='PyTorch Geometric ref (conda/pixi env)'),
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
