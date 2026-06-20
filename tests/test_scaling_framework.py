# -*- coding: utf-8 -*-
"""Scaling-extrapolation framework: the ``CostLaw`` / ``ScalePath``
declarations and the ``scaling_sweep`` param-point generator
(``cases/_base.py``).

The framework that lets a marquee op be measured across a dense range of small,
fast scales per modelling PATH and then *extrapolated* to brain scale
(combining the empirical fit with the theoretical ``CostLaw``).  Here we pin
the generator's shape contract; the fitter/projection math is pinned in
``test_extrapolate_report.py`` against synthetic rows with a planted exponent.
"""
from nperf.cases._base import CostLaw, ScalePath, scaling_sweep


def test_scaling_sweep_cross_product():
    """``scaling_sweep`` crosses each path with its own grid along its own
    ``cost.axis``, stamping the ``'path'`` label and merging the base + the
    path-fixing keys -- so two paths can sweep DIFFERENT axes in one case."""
    paths = [
        ScalePath('bin-many', {'family': 'binomial'},
                  CostLaw('q', 1.0, regime='many-tier q>64'), (16, 32, 64)),
        ScalePath('slope', {'family': 'gaussian', 'structure': 'unstructured'},
                  CostLaw('V', 1.0), (64, 256), challenging=True),
    ]
    pts = scaling_sweep(paths, base={'seed': 0})

    assert len(pts) == 3 + 2  # one point per (path, grid value)
    bin_pts = [p for p in pts if p['path'] == 'bin-many']
    slope_pts = [p for p in pts if p['path'] == 'slope']

    # each path swept its OWN axis (q vs V), at exactly its grid values
    assert sorted(p['q'] for p in bin_pts) == [16, 32, 64]
    assert sorted(p['V'] for p in slope_pts) == [64, 256]
    assert all('V' not in p for p in bin_pts)  # axes don't bleed across paths
    assert all('q' not in p for p in slope_pts)

    # base + path-fixing keys merged onto every point
    assert all(p['seed'] == 0 for p in pts)
    assert all(p['family'] == 'binomial' for p in bin_pts)
    assert all(p['structure'] == 'unstructured' for p in slope_pts)

    # the challenging flag is stamped only on the challenging path's rows
    assert all(p.get('challenging') for p in slope_pts)
    assert not any('challenging' in p for p in bin_pts)


def test_scaling_sweep_empty_base():
    """No ``base`` -> just path-fixing keys + the axis value + the label."""
    pts = scaling_sweep(
        [ScalePath('p', {'a': 1}, CostLaw('n', 2.0), (10, 100))])
    assert pts == [
        {'a': 1, 'n': 10, 'path': 'p'},
        {'a': 1, 'n': 100, 'path': 'p'},
    ]


def test_costlaw_is_frozen_hashable():
    """``CostLaw`` / ``ScalePath`` are frozen (safe to share across a case's
    declaration without aliasing surprises)."""
    cl = CostLaw('q', 3.0, hbm_exp=1.0, regime='few-tier dense')
    assert cl.axis == 'q' and cl.time_exp == 3.0
    assert hash(cl)  # frozen -> hashable
