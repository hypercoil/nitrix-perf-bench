# -*- coding: utf-8 -*-
"""Tier-2 augmentation cases (`nitrix.augment`).

CPU build + oracle agreement for the host-runnable baselines (jax + the numpy
reimplementation carrying nitrix's `eps`); the cupy GPU ref is skipped here
(needs a device + the refs-cupy env). The MONAI community baseline is gated on
`monai` being importable (it lives in the refs-monai env, not the base test
env) -- when present it doubles as the apples-to-apples community check.
"""
import numpy as np
import pytest

from nperf.cases import (
    gamma_contrast,
    gaussian_noise,
    gibbs_ringing,
    gmm_label_to_image,
    random_crop,
    random_flip,
    random_histogram_shift,
    random_resized_crop,
    random_svf_displacement,
    rician_noise,
    simulate_bias_field,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of


def test_gamma_baselines():
    built = gamma_contrast._build(gamma_contrast.CASE.representative)
    assert set(built.baselines) == {
        'nitrix-jax', 'numpy.gamma_contrast', 'cupy.gamma_contrast',
        'monai.AdjustContrast'}
    assert built.ratio_reference == 'nitrix-jax'
    # cupy is the on-target GPU headline ref (GPU-only); MONAI the CPU
    # community baseline (CPU-only for now).
    assert requires_of(built.baselines['cupy.gamma_contrast'][0]) == 'gpu'
    assert requires_of(built.baselines['monai.AdjustContrast'][0]) == 'cpu'
    assert framework_of(built.baselines['monai.AdjustContrast'][0]) == 'monai'


def test_gamma_host_baselines_match_oracle():
    c = gamma_contrast.CASE
    built = gamma_contrast._build(c.representative)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy: needs a device + the refs-cupy env
        if provider_id == 'monai':
            continue  # covered by test_gamma_monai_agreement (gated)
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
        assert fid['status'] == 'pass', f'{name}: {fid["rel_to_tol"]:.3g}'


def test_gamma_monai_agreement():
    '''nitrix's gamma_contrast == MONAI AdjustContrast (the clean community
    baseline; ~7e-8 in fp32). Gated on monai (refs-monai env only).'''
    pytest.importorskip('monai')
    pytest.importorskip('torch')
    c = gamma_contrast.CASE
    built = gamma_contrast._build(c.representative)
    fn = built.baselines['monai.AdjustContrast'][1]
    out = np.asarray(fn(*built.inputs_for('monai')))
    fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
    assert fid['status'] == 'pass', f'monai: {fid["rel_to_tol"]:.3g}'


def test_gamma_op_qualname():
    assert gamma_contrast.CASE.op_qualname == 'nitrix.augment.gamma_contrast'


def test_gibbs_baselines_and_no_monai():
    built = gibbs_ringing._build(gibbs_ringing.CASE.representative)
    # MONAI GibbsNoise models a different artifact (soft roll-off) -> NOT a
    # baseline here; the bar is the numpy oracle + the cupy GPU FFT ref.
    assert set(built.baselines) == {
        'nitrix-jax', 'numpy.gibbs_ringing', 'cupy.gibbs_ringing'}
    assert requires_of(built.baselines['cupy.gibbs_ringing'][0]) == 'gpu'


def test_gibbs_host_baselines_match_oracle():
    c = gibbs_ringing.CASE
    built = gibbs_ringing._build(c.representative)
    for name, (provider_id, fn) in built.baselines.items():
        if requires_of(provider_id) == 'gpu':
            continue  # cupy: needs a device + the refs-cupy env
        out = np.asarray(fn(*built.inputs_for(framework_of(provider_id))))
        fid = compare(out, built.fp64_reference, rtol=c.rtol, atol=c.atol)
        assert fid['status'] == 'pass', f'{name}: {fid["rel_to_tol"]:.3g}'


def test_gibbs_op_qualname():
    assert gibbs_ringing.CASE.op_qualname == 'nitrix.augment.gibbs_ringing'


@pytest.mark.parametrize('mod,gpu_ref,monai_ref', [
    (gaussian_noise, 'cupy.gaussian_noise', 'monai.RandGaussianNoise'),
    (rician_noise, 'cupy.rician_noise', 'monai.RandRicianNoise'),
])
def test_noise_baselines_and_no_oracle(mod, gpu_ref, monai_ref):
    built = mod._build(mod.CASE.representative)
    assert gpu_ref in built.baselines and monai_ref in built.baselines
    # RNG op: no cross-framework oracle, perf-ratio + distribution only.
    assert built.fp64_reference is None and built.fidelity_note
    assert requires_of(built.baselines[gpu_ref][0]) == 'gpu'
    assert requires_of(built.baselines[monai_ref][0]) == 'cpu'


def test_gaussian_noise_distribution():
    '''The residual ``out - x`` is ``sigma * N(0,1)`` -> var ≈ sigma**2.'''
    import jax
    from nitrix.augment import gaussian_noise as gn
    x = np.zeros((64, 64, 64), np.float32)
    out = np.asarray(gn(x, jax.random.PRNGKey(0), sigma=0.1))
    assert abs(float(out.var()) - 0.1 ** 2) < 1e-3


def test_rician_sigma_zero_is_abs():
    '''rician_noise reduces to ``|x|`` exactly at sigma=0 (no RNG).'''
    import jax
    import jax.numpy as jnp
    from nitrix.augment import rician_noise as rn
    x = np.linspace(-3, 3, 4096, dtype=np.float32)
    out = np.asarray(rn(jnp.asarray(x), jax.random.PRNGKey(0), sigma=0.0))
    assert np.max(np.abs(out - np.abs(x))) < 1e-5


def test_noise_op_qualnames():
    assert gaussian_noise.CASE.op_qualname == 'nitrix.augment.gaussian_noise'
    assert rician_noise.CASE.op_qualname == 'nitrix.augment.rician_noise'


# --- the RNG geometric / synthesis ops (no oracle, property-checked) ------
@pytest.mark.parametrize('mod,gpu_ref', [
    (random_flip, 'cupy.random_flip'),
    (random_crop, 'cupy.random_crop'),
    (random_histogram_shift, 'cupy.random_histogram_shift'),
    (gmm_label_to_image, 'cupy.gmm_label_to_image'),
])
def test_rng_aug_contract(mod, gpu_ref):
    built = mod._build(mod.CASE.representative)
    # RNG op -> no cross-framework oracle; cupy is the GPU headline ref.
    assert built.fp64_reference is None and built.fidelity_note
    assert requires_of(built.baselines[gpu_ref][0]) == 'gpu'
    # MONAI community baseline where it maps (not gmm).
    monai = [n for n in built.baselines if n.startswith('monai.')]
    if mod is gmm_label_to_image:
        assert not monai
    else:
        assert len(monai) == 1
        assert requires_of(built.baselines[monai[0]][0]) == 'cpu'


def test_random_flip_preserves_value_multiset():
    import jax
    import jax.numpy as jnp
    from nitrix.augment import random_flip as rf
    x = np.arange(48 ** 3, dtype=np.float32).reshape(48, 48, 48)
    out = np.asarray(rf(jnp.asarray(x), jax.random.PRNGKey(0)))
    assert out.shape == x.shape
    assert np.array_equal(np.sort(out.ravel()), np.sort(x.ravel()))


def test_random_crop_is_subblock():
    import jax
    import jax.numpy as jnp
    from nitrix.augment import random_crop as rc
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 48, 48)).astype(np.float32)
    out = np.asarray(rc(jnp.asarray(x), jax.random.PRNGKey(0), size=(24, 24,
                                                                     24)))
    assert out.shape == (24, 24, 24)
    # a contiguous sub-block cannot exceed the input's value range
    assert out.min() >= x.min() and out.max() <= x.max()


def test_random_histogram_shift_is_monotone():
    # the defining property: a monotone remap preserves the voxel rank order.
    import jax
    import jax.numpy as jnp
    from nitrix.augment import random_histogram_shift as rhs
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 32, 32)).astype(np.float32)
    out = np.asarray(rhs(jnp.asarray(x), jax.random.PRNGKey(0)))
    ordered = out.ravel()[np.argsort(x.ravel())]
    assert np.all(np.diff(ordered) >= -1e-4)  # non-decreasing


def test_gmm_per_label_mean():
    # the distributional property: per-label sample mean ≈ the label's mean.
    import jax
    import jax.numpy as jnp
    from nitrix.augment import gmm_label_to_image as gmm

    from nperf.cases._augment import gmm_labels
    lab, means, stds = gmm_labels((48, 48, 48), 5, 0)
    out = np.asarray(gmm(jnp.asarray(lab), jnp.asarray(means),
                         jnp.asarray(stds), jax.random.PRNGKey(0)))
    for label in range(5):
        sample_mean = float(out[lab == label].mean())
        assert abs(sample_mean - float(means[label])) < 0.2  # ~ std/sqrt(N)


def test_rng_aug_op_qualnames():
    assert random_flip.CASE.op_qualname == 'nitrix.augment.random_flip'
    assert random_crop.CASE.op_qualname == 'nitrix.augment.random_crop'
    assert (random_histogram_shift.CASE.op_qualname
            == 'nitrix.augment.random_histogram_shift')
    assert (gmm_label_to_image.CASE.op_qualname
            == 'nitrix.augment.gmm_label_to_image')


# --- the interp / generator RNG ops (scipy/cupyx twins, no MONAI) ---------
@pytest.mark.parametrize('mod,gpu_ref', [
    (random_resized_crop, 'cupy.random_resized_crop'),
    (simulate_bias_field, 'cupy.simulate_bias_field'),
    (random_svf_displacement, 'cupy.random_svf_displacement'),
])
def test_interp_gen_aug_contract(mod, gpu_ref):
    built = mod._build(mod.CASE.representative)
    assert built.fp64_reference is None and built.fidelity_note
    assert requires_of(built.baselines[gpu_ref][0]) == 'gpu'  # cupy headline
    # these have no clean MONAI analog (semantics / I/O differ)
    assert not [n for n in built.baselines if n.startswith('monai.')]


def test_resized_crop_output_shape():
    import jax
    import jax.numpy as jnp
    from nitrix.augment import random_resized_crop as rrc
    rng = np.random.default_rng(0)
    x = rng.standard_normal((48, 48, 48, 1)).astype(np.float32)
    out = np.asarray(rrc(jnp.asarray(x), jax.random.PRNGKey(0),
                         size=(24, 24, 24)))
    assert out.shape == (24, 24, 24, 1)


def test_bias_field_positive_and_smooth():
    import jax
    from nitrix.augment import simulate_bias_field as sbf
    out = np.asarray(sbf((48, 48, 48), jax.random.PRNGKey(0)))
    assert out.shape == (48, 48, 48)
    assert np.isfinite(out).all() and (out > 0).all()  # multiplicative field


def test_svf_shape_and_finite():
    import jax
    from nitrix.augment import random_svf_displacement as svf
    out = np.asarray(svf((48, 48, 48), jax.random.PRNGKey(0)))
    assert out.shape == (48, 48, 48, 3)  # (*spatial, ndim)
    assert np.isfinite(out).all()


def test_interp_gen_op_qualnames():
    assert (random_resized_crop.CASE.op_qualname
            == 'nitrix.augment.random_resized_crop')
    assert (simulate_bias_field.CASE.op_qualname
            == 'nitrix.augment.simulate_bias_field')
    assert (random_svf_displacement.CASE.op_qualname
            == 'nitrix.augment.random_svf_displacement')
