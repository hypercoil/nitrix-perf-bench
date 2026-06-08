# -*- coding: utf-8 -*-
"""Shared helpers for the grayscale-morphology family (erode / dilate / open /
close) -- built so the comparison is *warranted* on the exact thing nitrix
computes.

**The B18 Win 3 seam.** nitrix's flat-box default (``size=`` /
``structuring_element=None``) lowers to a fused ``lax.reduce_window`` -- the
fast path.  But **any explicit structuring element -- including a flat disk /
ball footprint, the default footprint in skimage and the common choice in
scipy -- routes through the slow ``semiring_conv`` (im2col + tropical-matmul)
path.**  A box-only bench certifies "fast morphology" while the footprint users
actually pick is on the slow branch; these cases measure *both* SE shapes as
param points so that gap is visible.

**SE-encoding contract** (verified vs scipy: box/3D exact, disk/grayscale to
~5e-8 fp32 round-off -- see ``tests/test_morphology_cases.py``):

- nitrix's ``structuring_element`` is the *additive* grayscale structure
  (``dilate: out[i] = max_p(x[i+p] + se[p])``; scipy's ``structure=``), **not**
  a boolean footprint.  A flat footprint ``F`` (reduce over the ``F``-True
  neighbours only) is therefore encoded ``se = where(F, 0.0, -inf)``: the
  ``-inf`` outside ``F`` drops those positions from the max (and, via erosion's
  ``x - se``, from the min).  Passing a binary ``{0,1}`` mask would be *wrong*:
  the ``0`` region still participates with a ``+0`` offset (a full box with a
  disk-shaped bump, not a disk).
- nitrix's border is SAME + the tropical identity (``-inf`` for max-plus,
  ``+inf`` for min-plus): out-of-bounds neighbours are *ignored*.  The matching
  scipy / cupy oracle is therefore ``mode='constant', cval=-/+inf``.  For a
  flat box this coincides with scipy's default ``reflect`` (reflection only
  duplicates an already-included neighbour, and min/max are idempotent), which
  is why the old box-only case matched; for a disk it does **not**, so we pin
  the constant form that provably matches nitrix's border (by construction;
  B13).

scipy.ndimage is a core dep (top-level); cupy is lazy (refs-cupy worker only);
SimpleITK is the flat-box floor (main env), box only.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as spnd

# Tropical identities = the constant-pad value that makes scipy ignore OOB,
# matching nitrix's SAME + identity border (dilation/max -> -inf; erosion/min
# -> +inf).
_PAD = {'dilate': -np.inf, 'erode': np.inf}

_DTYPE = {'float32': np.float32, 'float16': np.float16}


def morph_input(shape, seed: int = 0, dtype: str = 'float32') -> np.ndarray:
    '''A random image, the dtype the case measures (fp32 default, fp16 row).'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal(tuple(shape)).astype(_DTYPE[dtype])


def disk_footprint(radius: int, rank: int) -> np.ndarray:
    '''Centro-symmetric disk (2-D) / ball (3-D) boolean footprint -- the
    default skimage footprint and the common scipy choice.  Symmetric, so
    scipy's dilation footprint-reflection is a no-op and the comparison needs
    no origin bookkeeping.'''
    a = np.arange(-radius, radius + 1)
    grids = np.meshgrid(*([a] * rank), indexing='ij')
    r2 = sum(g.astype(np.float64) ** 2 for g in grids)
    return r2 <= float(radius) ** 2


def resolve_se(
    param: Dict[str, Any], dtype: str,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    '''Map a param point to ``(se_spec, se_array)``.

    ``se_spec`` drives the scipy / cupy refs (``{'kind': 'box', 'size': k}`` or
    ``{'kind': <'disk'|'ball'>, 'footprint': F}``); ``se_array`` is nitrix's
    additive structuring element (``None`` for the flat-box fast path, which
    nitrix selects via ``size=``).'''
    shape = param['shape']
    if param['se'] == 'box':
        return {'kind': 'box', 'size': int(param['size'])}, None
    fp = disk_footprint(int(param['radius']), len(shape))
    se = np.where(fp, 0.0, -np.inf).astype(_DTYPE[dtype])
    return {'kind': param['se'], 'footprint': fp}, se


def nitrix_kwargs(
    se_spec: Dict[str, Any], se_array: Any,
) -> Dict[str, Any]:
    '''Kwargs for a nitrix morphology op: ``size=`` (flat-box fast path) or
    ``structuring_element=`` (explicit SE -> slow semiring path).'''
    if se_spec['kind'] == 'box':
        return {'size': se_spec['size']}
    return {'structuring_element': se_array}


def _erode_dilate(name: str, x: Any, se_spec: Dict[str, Any], cval: float,
                  module: Any) -> Any:
    fn = getattr(module, name)
    if se_spec['kind'] == 'box':
        return fn(x, size=se_spec['size'], mode='constant', cval=cval)
    fp = se_spec['footprint']
    if module is not spnd:  # cupy: footprint must be on-device
        import cupy as cp

        fp = cp.asarray(fp)
    return fn(x, footprint=fp, mode='constant', cval=cval)


def _morph(kind: str, x: Any, se_spec: Dict[str, Any], module: Any) -> Any:
    '''Compute one morphology op via grey_erosion / grey_dilation, with the
    constant-pad cval that matches nitrix's border.  open / close are composed
    explicitly (erode-then-dilate / dilate-then-erode) so *each* pass gets its
    own matching cval -- scipy's single-``cval`` ``grey_opening`` cannot match
    nitrix's two different identities at the border.'''
    de, di = _PAD['erode'], _PAD['dilate']
    if kind == 'dilate':
        return _erode_dilate('grey_dilation', x, se_spec, di, module)
    if kind == 'erode':
        return _erode_dilate('grey_erosion', x, se_spec, de, module)
    if kind == 'open':  # erode then dilate
        e = _erode_dilate('grey_erosion', x, se_spec, de, module)
        return _erode_dilate('grey_dilation', e, se_spec, di, module)
    if kind == 'close':  # dilate then erode
        d = _erode_dilate('grey_dilation', x, se_spec, di, module)
        return _erode_dilate('grey_erosion', d, se_spec, de, module)
    raise ValueError(f'unknown morphology kind {kind!r}')


def scipy_morph(kind: str, se_spec: Dict[str, Any]) -> Callable[[Any], Any]:
    '''scipy.ndimage reference / fp64 oracle for one morphology op (CPU floor).
    Border pinned to ``mode='constant', cval=±inf`` so it is the *same* op
    nitrix computes (B13), not scipy's default reflect.'''
    return lambda x: _morph(kind, np.asarray(x), se_spec, spnd)


def cupy_morph(kind: str, se_spec: Dict[str, Any]) -> Callable[[Any], Any]:
    '''CuPy ``cupyx.scipy.ndimage`` on-target GPU reference; cupy is lazy
    (imported only in the refs-cupy worker).'''

    def run(x: Any) -> Any:
        from cupyx.scipy import ndimage as cnd

        return _morph(kind, x, se_spec, cnd)

    return run


# ---------------------------------------------------------------------------
# Scale tier (B23 / scale-gaming): brain-scale single-volume + batched-cohort
# points.  The explicit-SE path's im2col cost compounds with grid size and
# batch toward an OOM the dev tier never reaches; the batched refs loop because
# scipy / cupy morphology require ``footprint.ndim == input.ndim`` (and the
# explicit-SE op is per-image: a rank-d SE on a rank-(d+1) stack windows ``1``
# on the leading batch axis, so nitrix consumes the stack directly).
# ---------------------------------------------------------------------------

_GREY = {'dilate': 'grey_dilation', 'erode': 'grey_erosion',
         'open': 'grey_opening', 'close': 'grey_closing'}


def morph_stack(batch: int, shape, seed: int = 0,
                dtype: str = 'float32') -> np.ndarray:
    '''A cohort stack ``(batch, *spatial)`` -- the batched brain-data regime
    where the per-volume im2col cost compounds toward OOM.'''
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch,) + tuple(shape)).astype(_DTYPE[dtype])


def scipy_morph_batched(kind: str,
                        se_spec: Dict[str, Any]) -> Callable[[Any], Any]:
    '''Per-image scipy morphology over a leading batch axis (the references
    must loop -- scipy/cupy need ``footprint.ndim == input.ndim`` and the op is
    per-image).'''
    one = scipy_morph(kind, se_spec)

    def run(x: Any) -> Any:
        a = np.asarray(x)
        return np.stack([one(a[i]) for i in range(a.shape[0])])

    return run


def cupy_morph_batched(kind: str,
                       se_spec: Dict[str, Any]) -> Callable[[Any], Any]:
    '''Per-image CuPy morphology over a leading batch axis; cupy lazy.'''

    def run(x: Any) -> Any:
        import cupy as cp

        one = cupy_morph(kind, se_spec)
        return cp.stack([one(x[i]) for i in range(x.shape[0])])

    return run


def build_morph_large(kind: str, nitrix_op: Callable[..., Any],
                      param: Dict[str, Any]) -> Any:
    '''A ``BuiltPoint`` for a brain-scale **size-tier** point (shared by all
    four morphology cases).  Baselines are **nitrix + the cupy GPU ref only**,
    and there is **no fp64 oracle** (``fp64_reference=None`` -> fidelity
    inconclusive): correctness is pinned tight at the dev tier (the op is exact
    and the code path is identical), so this tier measures *scale* -- speed,
    HBM, and OOM -- where an O(N*k) CPU oracle / floor at 256^3 would be both
    slow and beside the point.  Single large volumes and batched cohorts.'''
    import jax
    import jax.numpy as jnp

    from ._base import BuiltPoint, to_cupy

    dtype = param.get('dtype', 'float32')
    batch = param.get('batch')
    se_spec, se = resolve_se(param, dtype)
    se_jax = None if se is None else jnp.asarray(se)
    kw = nitrix_kwargs(se_spec, se_jax)

    if batch:
        X = morph_stack(batch, param['shape'], param.get('seed', 0), dtype)
        cupy_fn = cupy_morph_batched(kind, se_spec)
    else:
        X = morph_input(param['shape'], param.get('seed', 0), dtype)
        cupy_fn = cupy_morph(kind, se_spec)
    jx = jax.block_until_ready(jnp.asarray(X))

    def inputs_for(framework: str):
        if framework == 'cupy':
            return to_cupy(X)
        return (X,) if framework == 'numpy' else (jx,)

    baselines = {
        'nitrix-jax': ('jax', lambda x: nitrix_op(x, **kw)),
        f'cupyx.scipy.ndimage.{_GREY[kind]}': ('cupy', cupy_fn),
    }
    return BuiltPoint(
        baselines=baselines, inputs_for=inputs_for, fp64_reference=None,
        fidelity_note=('scale tier: fidelity pinned at the dev tier; this '
                       'point measures scale (speed / HBM / OOM)'),
        ratio_reference='nitrix-jax',
    )
