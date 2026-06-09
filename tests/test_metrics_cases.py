# -*- coding: utf-8 -*-
"""Registration metric cases (R0): the similarity-score family.

CPU build + oracle agreement for the host baselines (nitrix + the numpy
exact-convention reimplementation + the domain references). Pins the verified
convention relationships (2026-06-09): ssd == ITK ``MeanSquares`` (a
co-oracle); ncc == signed Pearson, ITK ``Correlation`` == ``-ncc**2``; lncc ==
ANTs ``ANTSNeighborhoodCorrelation`` interior (sign + boundary); MI order-1
Parzen vs ITK Mattes (order-3) / sklearn (order-0); CR has no domain co-oracle.
The divergent domain tools ride as labelled ``ApproxBaseline``s (not gated).
cupy GPU refs are skipped here.
"""
import numpy as np
import pytest

from nperf.cases import (
    correlation_ratio,
    lncc,
    mutual_information,
    ncc,
    ssd,
)
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of

_MODS = [ssd, ncc, lncc, mutual_information, correlation_ratio]


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_ratio_reference_and_qualname(mod):
    built = mod._build(mod.CASE.representative)
    assert built.ratio_reference == 'nitrix-jax'
    assert mod.CASE.op_qualname == f'nitrix.metrics.{mod.CASE.name}'
    # every metric ships a numpy oracle + a cupy GPU bar.
    provs = {pid for pid, _ in built.baselines.values()}
    assert {'jax', 'numpy', 'cupy'} <= provs


@pytest.mark.parametrize('mod', _MODS, ids=lambda m: m.CASE.name)
def test_host_baselines_match_oracle(mod):
    # Gated host baselines (not gpu, not declared-approximate) must match the
    # fp64 oracle; the divergent domain tools are declared ApproxBaseline and
    # skipped here (their divergence is the point, verified separately).
    case = mod.CASE
    approx = {a.baseline for a in case.approximate_baselines}
    for p in case.param_points:
        built = mod._build(p)
        for name, (pid, fn) in built.baselines.items():
            if requires_of(pid) == 'gpu' or name in approx:
                continue
            out = np.asarray(fn(*built.inputs_for(framework_of(pid))))
            fid = compare(out, built.fp64_reference,
                          rtol=case.rtol, atol=case.atol)
            assert fid['status'] == 'pass', (
                f'{case.name}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}')


def test_ncc_itk_divergence_is_the_documented_transform():
    # ITK Correlation = -(signed Pearson r)**2 -- why simpleitk.Correlation is
    # a labelled-divergent ApproxBaseline, not a co-oracle.
    built = ncc._build(ncc.CASE.representative)
    r = float(np.asarray(built.baselines['nitrix-jax'][1](
        *built.inputs_for('jax'))))
    itk = float(np.asarray(built.baselines['simpleitk.Correlation'][1](
        *built.inputs_for('numpy'))))
    assert abs(itk - (-r ** 2)) < 1e-3


def test_divergent_domain_refs_are_declared_approximate():
    # The domain tools that compute a different quantity are declared, so the
    # gate never fails on them (fidelity reported, not gated).
    assert 'simpleitk.Correlation' in _approx(ncc)
    assert 'simpleitk.ANTSNeighborhoodCorrelation' in _approx(lncc)
    assert {'simpleitk.MattesMI', 'sklearn.mutual_info'} <= _approx(
        mutual_information)
    # CR has no domain co-oracle -> no divergent ref to declare.
    assert _approx(correlation_ratio) == set()


def _approx(mod):
    return {a.baseline for a in mod.CASE.approximate_baselines}
