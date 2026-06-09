# -*- coding: utf-8 -*-
"""Hardened morphology cases (B18 Win 3): erode / dilate / open / close.

The flat-box default lowers to a fused ``lax.reduce_window`` (the fast path),
but **any explicit structuring element -- including a flat disk / ball
footprint, the default footprint users actually pick -- routes through the slow
``semiring_conv`` path.**  A box-only bench would hide that.  These checks pin
the four seams the perf rows ride on, so a perf win on the fast box branch can
never land while a correctness / capability contract on the disk branch is red:

- **border parity (B13)**: nitrix's SAME + ``±inf`` identity is the *same* op
  as scipy ``mode='constant', cval=±inf`` -- asserted across the whole array
  (borders included) so a fast path that quietly changed the border fails here.
- **fast / slow paths agree**: a flat box passed *as an explicit SE* (the slow
  semiring path) equals the ``size=`` fast path -- the perf gap is pure
  dispatch overhead, not a different answer.
- **SE-encoding contract**: a flat footprint must be the *additive* ``-inf``
  encoding, not a binary ``{0,1}`` mask (which would add a disk-shaped bump).
- **jit(grad) capability (B19)**: the flat-box ``reduce_window`` fast path must
  differentiate under ``jit(grad(...))`` (the double-transform training loops
  run) -- the regression that paused this work, now gated, not just timed.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nitrix.morphology import close as nx_close
from nitrix.morphology import dilate as nx_dilate
from nitrix.morphology import erode as nx_erode
from nitrix.morphology import open as nx_open

from nperf.cases import close as closing
from nperf.cases import dilate, erode
from nperf.cases import open as opening
from nperf.cases._morphology import disk_footprint, scipy_morph
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_NX = {'dilate': nx_dilate, 'erode': nx_erode,
       'open': nx_open, 'close': nx_close}
_MODS = {'dilate': dilate, 'erode': erode, 'open': opening, 'close': closing}


@pytest.mark.parametrize('mod', list(_MODS.values()))
def test_oracle_match_tight_all_points(mod):
    # Every CPU baseline (scipy floor, ITK box floor, nitrix itself) must pass
    # the tight gate at every param point -- including the borders, where the
    # constant-pad oracle pins nitrix's SAME + identity convention (B13).
    case = mod.CASE
    for param in case.param_points:
        built = case.build(param)
        for name, (pid, fn) in built.baselines.items():
            if requires_of(pid) == 'gpu':
                continue  # cupy ref needs a device
            out = np.asarray(fn(*built.inputs_for(framework_of(pid))),
                             dtype=np.float64)
            fid = compare(out, built.fp64_reference,
                          rtol=case.rtol, atol=case.atol)
            assert fid['status'] == 'pass', (
                f'{case.name}/{name}@{param}: '
                f'rel_to_tol={fid["rel_to_tol"]:.3g}'
            )


@pytest.mark.parametrize('kind', ['dilate', 'erode'])
def test_fast_and_slow_paths_agree(kind):
    # The flat-box fast path (``size=``, reduce_window) and the same flat box
    # routed through the slow semiring path (an all-zero explicit SE) must give
    # the *same* answer -- so the disk/box perf gap is pure dispatch overhead,
    # not a semantic difference.
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((40, 40)).astype(np.float32))
    fn = _NX[kind]
    fast = np.asarray(fn(x, size=3))  # reduce_window fast path
    box_se = jnp.zeros((3, 3))         # all-zero flat box -> semiring path
    slow = np.asarray(fn(x, structuring_element=box_se))
    assert np.abs(fast - slow).max() < 1e-4


@pytest.mark.parametrize('kind', ['dilate', 'erode'])
def test_disk_se_encoding_is_additive_not_binary(kind):
    # The flat-footprint SE is the *additive* encoding (0 inside, -inf out);
    # a binary {0,1} disk is a different op (it adds a +1 bump inside the disk,
    # i.e. a full box, not a disk).  Guards _morphology's flat-SE construction.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 40)).astype(np.float64)
    D = disk_footprint(3, 2)
    se_flat = jnp.asarray(np.where(D, 0.0, -np.inf))
    se_binary = jnp.asarray(D.astype(np.float64))
    fn = _NX[kind]
    nx_flat = np.asarray(fn(jnp.asarray(X), structuring_element=se_flat))
    nx_binary = np.asarray(fn(jnp.asarray(X), structuring_element=se_binary))
    ref = scipy_morph(kind, {'kind': 'disk', 'footprint': D})(X)
    assert np.abs(nx_flat - ref).max() < 1e-3     # correct encoding
    assert np.abs(nx_binary - ref).max() > 0.1    # binary is a different op


@pytest.mark.parametrize('kind', ['dilate', 'erode', 'open', 'close'])
@pytest.mark.parametrize('se', ['box', 'disk'])
def test_jitgrad_composes_and_matches_eager(kind, se):
    # B19 gate: the flat-box reduce_window fast path must differentiate under
    # jit(grad(...)) (it regressed once -- a traced window init fell off the
    # differentiable monoid primitive).  The semiring disk path always did;
    # both are pinned here so a perf win can't reintroduce the regression.
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal((16, 16)).astype(np.float32))
    fn = _NX[kind]
    if se == 'box':
        kw = {'size': 3}
    else:
        D = disk_footprint(2, 2)
        kw = {'structuring_element':
              jnp.asarray(np.where(D, 0.0, -np.inf).astype(np.float32))}

    def loss(z):
        return jnp.sum(fn(z, **kw) ** 2)

    g_eager = np.asarray(jax.grad(loss)(x))
    g_jit = np.asarray(jax.jit(jax.grad(loss))(x))
    assert np.isfinite(g_jit).all()                # no -inf/nan leak
    assert np.allclose(g_eager, g_jit, atol=1e-6)  # jit(grad) == eager
    assert (g_jit != 0).any()                      # a real gradient


def test_batch_contract_int_size_leaks_tuple_does_not():
    # Morphology contract (mirrors distance_transform): an *int* size treats
    # every axis as spatial, so on a stack it leaks across the batch; a *tuple*
    # size (or an explicit SE) pins the spatial rank and is per-image, as is
    # vmap.  Asserted so the gotcha is visible, not silent.
    rng = np.random.default_rng(3)
    stack = jnp.asarray(rng.standard_normal((4, 24, 24)).astype(np.float32))
    per = np.stack([np.asarray(nx_dilate(stack[i], size=(3, 3)))
                    for i in range(4)])
    tup = np.asarray(nx_dilate(stack, size=(3, 3)))       # spatial_rank=2
    leak = np.asarray(nx_dilate(stack, size=3))           # all-axes-spatial
    vm = np.asarray(jax.vmap(lambda z: nx_dilate(z, size=3))(stack))
    assert np.abs(tup - per).max() < 1e-4                 # tuple == per-image
    assert np.abs(vm - per).max() < 1e-4                  # vmap == per-image
    assert np.abs(leak - per).max() > 0.1                 # int size leaks


@pytest.mark.parametrize('kind', ['dilate', 'erode'])
def test_fp16_is_exact(kind):
    # min/max picks an input value verbatim, so the fp16 path is exact vs the
    # fp64 oracle (the case's fp16 row is a precision/perf probe, nitrix-only;
    # scipy/cupy have no native fp16 morphology).
    rng = np.random.default_rng(4)
    Xh = rng.standard_normal((40, 40)).astype(np.float16)
    fn = _NX[kind]
    out = fn(jnp.asarray(Xh), size=3)
    assert out.dtype == jnp.float16
    ref = scipy_morph(kind, {'kind': 'box', 'size': 3})(Xh.astype(np.float64))
    assert np.abs(np.asarray(out, np.float64) - ref).max() < 1e-3


def test_op_qualnames():
    assert dilate.CASE.op_qualname == 'nitrix.morphology.dilate'
    assert erode.CASE.op_qualname == 'nitrix.morphology.erode'
    assert opening.CASE.op_qualname == 'nitrix.morphology.open'
    assert closing.CASE.op_qualname == 'nitrix.morphology.close'
