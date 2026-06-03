# -*- coding: utf-8 -*-
"""Tier-2 bilateral case (bilateral_gaussian as a grid image bilateral).

No fp64 oracle: nitrix's bilateral matches ITK's ``BilateralImageFilter`` in
the **interior** (the bounded window + domain/range Gaussians + normalisation
all match), but the r-pixel boundary diverges (ITK edge vs replicate stencil).
So the load-bearing check is interior parity with SimpleITK -- the no-oracle
analogue of the median_filter interior check.
"""
import math

import numpy as np
import pytest

from nperf.cases import bilateral_gaussian as bg
from nperf.providers import framework_of, requires_of


def test_baselines_and_no_oracle():
    built = bg._build({'shape': [48, 48], 'sigma_d': 2.0, 'sigma_r': 0.2,
                       'seed': 0})
    assert set(built.baselines) == {'nitrix-jax', 'simpleitk.Bilateral'}
    assert built.ratio_reference == 'nitrix-jax'
    assert built.fp64_reference is None and built.fidelity_note
    assert framework_of(built.baselines['simpleitk.Bilateral'][0]) == 'numpy'
    assert requires_of(built.baselines['simpleitk.Bilateral'][0]) is None


def test_interior_parity_with_itk():
    '''The correctness gate: nitrix's grid bilateral matches sitk.Bilateral in
    the interior (away from the r-pixel boundary).'''
    pytest.importorskip('SimpleITK')
    h = w = 48
    sd, sr = 2.0, 0.2
    built = bg._build({'shape': [h, w], 'sigma_d': sd, 'sigma_r': sr,
                       'seed': 0})
    (img,) = built.inputs_for('numpy')
    nit = np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax'))).reshape(h, w)
    itk = np.asarray(built.baselines['simpleitk.Bilateral'][1](img))
    r = math.ceil(2.5 * sd)
    interior_err = np.max(np.abs((nit - itk)[r:-r, r:-r]))
    assert interior_err < 1e-3, f'interior parity {interior_err:.2e}'


def test_op_qualname():
    assert bg.CASE.op_qualname == 'nitrix.smoothing.bilateral_gaussian'
