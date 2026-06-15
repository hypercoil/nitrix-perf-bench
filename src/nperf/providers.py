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
        # nilearn reference (nilearn.connectome -- the canonical neuroimaging
        # CPU floor for connectome ops like the tangent-space embedding): host,
        # numpy framework + base env, same shape as sklearn/scipy.  A distinct
        # id so the floor is attributable to the domain-standard tool (the
        # first domain-tool reference; see DOMAIN_TOOL_BASELINES.md).
        Provider('nilearn', 'numpy', description='host nilearn (uv)'),
        # networkx reference (the canonical graph library -- modularity matrix
        # etc.): host, numpy framework + base env, lazy-imported in-case.
        Provider('networkx', 'numpy', description='host networkx (uv)'),
        # SimpleITK reference (ITK N4 bias correction + HistogramMatching --
        # the canonical medical-imaging tools nitrix.bias parity-tests
        # against): host, numpy framework + base env, distinct id for
        # attribution.  Lazy-imported in-case (only this provider's worker
        # needs it).  See DOMAIN_TOOL_BASELINES.md (Tier 1).
        Provider('simpleitk', 'numpy', description='host SimpleITK (uv)'),
        # statsmodels reference (statsmodels.MixedLM -- the canonical CPU LME
        # tool): host, numpy framework.  requires='cpu' so it runs ONLY on the
        # CPU platform -- it has no GPU path and is *expensive* (per-voxel
        # iterative fits), so running identical host work under the GPU
        # platform label would just double the cost; the headline GPU-vs-CPU
        # speedup is read cross-platform from nitrix's two rows + this one.
        Provider('statsmodels', 'numpy', requires='cpu',
                 description='host statsmodels MixedLM (uv); CPU-only ref'),
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
        # ANTsPy reference (the canonical medical-imaging registration /
        # resampling tool; ITK-backed).  Its antspyx wheel pins its own
        # scipy/sklearn, so it lives in an isolated refs env (DESIGN §7) like
        # torch -- framework 'ants', selected per attempt via
        # NPERF_PYTHON_ANTS; CPU tool, lazy-imported in-case.
        # requires='cpu' (like statsmodels): a CPU-only tool whose refs env
        # ships CPU-only jax, so it can't satisfy the jax-cuda12 platform's
        # JAX_PLATFORMS=cuda -- it runs on jax-cpu only, and nitrix's GPU-vs-
        # ANTs comparison is read cross-platform from nitrix's two rows.
        Provider('ants', 'ants', requires='cpu',
                 description='ANTsPy ref (resample/transforms); refs-ants env '
                             '(NPERF_PYTHON_ANTS); CPU-only'),
        # dipy reference (the numpy/scipy/cython registration toolkit -- the
        # second cross-tool registration foil alongside ANTs).  Pure-Python /
        # cython CPU tool returning host numpy arrays (framework 'dipy' -> the
        # numpy no-op sync); its own refs env (it pins its own numpy/scipy),
        # selected per attempt via NPERF_PYTHON_DIPY; lazy-imported in-case.
        # requires='cpu' (like ants/statsmodels): no GPU path, and its refs env
        # ships CPU-only jax, so it runs on jax-cpu only -- the nitrix-GPU-vs-
        # dipy comparison is read cross-platform from nitrix's two rows.
        Provider('dipy', 'dipy', requires='cpu',
                 description='dipy registration ref (numpy/scipy/cython); '
                             'refs-dipy env (NPERF_PYTHON_DIPY); CPU-only'),
        # AFNI / FSL -- the *community* registration tools (3dvolreg / mcflirt
        # for motion realignment; flirt -bbr for BBR). Unlike ants/dipy these
        # are command-line BINARIES, not Python packages, so they need no
        # separate interpreter: framework 'numpy' (the base env -- which has
        # nibabel for the NIfTI round-trip + nitrix for the lazy case import),
        # and the wrapper shells out to the binary located via NPERF_AFNI_DIR /
        # NPERF_FSL_DIR (absolute dir, not $PATH). requires='cpu' (like
        # statsmodels/ants): a CPU tool -> runs on jax-cpu only; the
        # nitrix-GPU-vs-AFNI/FSL economic comparison is read cross-platform.
        # Spin-up: tools/setup_neuro_refs.sh (see README "Community neuro
        # reference tools"); /scratch is ephemeral, that script is the recipe.
        # MONAI reference (the de-facto community medical-imaging augmentation
        # / metric toolkit; torch-backed).  Its own refs env (jax+nitrix+torch+
        # monai), selected per attempt via NPERF_PYTHON_MONAI; lazy-imported
        # in-case.  framework 'monai' -> the numpy no-op sync (the wrappers
        # force CPU torch tensors, which are synchronous; HBM falls to None as
        # for the other CPU refs).  requires='cpu' FOR NOW: a CPU community
        # baseline (the GPU headline ref stays cupy/torch); a GPU MONAI env
        # (jax-cuda + torch-cuda co-install) is a tracked follow-up for the
        # tests that need MONAI on-device. See monai-augmentation-parity FR.
        Provider('monai', 'monai', requires='cpu',
                 description='MONAI ref (transforms/metrics); refs-monai env '
                             '(NPERF_PYTHON_MONAI); CPU community baseline '
                             '(GPU follow-up pending)'),
        Provider('afni', 'numpy', requires='cpu',
                 description='AFNI ref (3dvolreg realign; 3dcalc io-floor); '
                             'binary at NPERF_AFNI_DIR; CPU-only'),
        Provider('fsl', 'numpy', requires='cpu',
                 description='FSL ref (mcflirt realign; fslmaths io-floor); '
                             'binary at NPERF_FSL_DIR; CPU-only'),
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
