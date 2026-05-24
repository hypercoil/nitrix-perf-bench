# -*- coding: utf-8 -*-
"""The HTML /site renderer (report/html.py).

A self-contained page: current-state tables + inline-SVG plots, derived from L4
rows (no hand-edited values).  These are smoke + structure checks — the page is
git-ignored, so the renderer (not the output) is what's tested.
"""
from nperf.report import render_site
from nperf.report.html import _size, _svg_loglog


def _row(case, platform, baseline, mn, size_n, *, status='ok', run_id='r1'):
    r = {
        'case': case, 'platform': platform, 'baseline': baseline,
        'status': status, 'run_id': run_id,
        'param_point': {'n': size_n, 'k': 16, 'seed': 0},
        'provenance': {'nitrix': {'sha': 'abc1234567'},
                       'bench': {'sha': 'def7654321'},
                       'os': 'linux', 'jax_version': '0.10.0'},
    }
    if status == 'ok':
        r['metrics'] = {'steady_time': {'min': mn, 'median': mn, 'p95': mn,
                                        'unit': 's'}}
        if baseline != 'nitrix-jax':
            r['ratio'] = {'vs': 'nitrix-jax', 'metric': 'min', 'value': 0.5}
        r['fidelity'] = {'status': 'pass', 'rel_to_tol': 0.1}
    return r


def test_size_is_product_excluding_seed():
    assert _size({'n': 16384, 'k': 16, 'seed': 0}) == 16384 * 16
    assert _size({'m': 512, 'k': 512, 'n': 512, 'algebra': 'log'}) == 512 ** 3


def test_svg_handles_empty_and_degenerate():
    assert 'no plottable data' in _svg_loglog({}, xlabel='x', ylabel='y')
    # single point (degenerate ranges) must still produce an <svg>, not crash.
    svg = _svg_loglog({'s': [(1.0, 1.0)]}, xlabel='x', ylabel='y')
    assert svg.startswith('<svg') and '</svg>' in svg


def test_site_is_self_contained_and_covers_cases():
    rows = [
        _row('semiring_matmul', 'jax-cpu', 'nitrix-jax', 1e-3, 256),
        _row('semiring_matmul', 'jax-cpu', 'pyg', 5e-4, 256),
        _row('semiring_matmul', 'jax-cpu', 'nitrix-jax', 4e-3, 512),
        _row('ell_edge_aggregate', 'jax-cuda12', 'nitrix-jax', 8e-4, 16384),
    ]
    html = render_site(rows)
    # one self-contained doc: style + script inlined, no external src/href.
    assert html.startswith('<!doctype html>')
    assert '<style>' in html and '<script>' in html
    assert 'src=' not in html and 'http' not in html
    # both cases present, with an inline plot each.
    assert 'semiring_matmul' in html and 'ell_edge_aggregate' in html
    assert html.count('<svg') >= 2
    # provenance + the no-hand-edit provenance note.
    assert 'no hand-edited values' in html and 'def765' in html


def test_caveats_footer_always_present():
    rows = [_row('semiring_matmul', 'jax-cpu', 'nitrix-jax', 1e-3, 256)]
    assert 'How to read these numbers' in render_site(rows)


def test_capability_overlay_links_benchmarked_ops():
    rows = [_row('semiring_matmul', 'jax-cpu', 'nitrix-jax', 1e-3, 256)]
    capability = {
        'host': {'device': 'NVIDIA A10G (gpu)'},
        'ops': [
            # benchmarked here -> should get a ⚡ link to #semiring_matmul
            {'qualname': 'nitrix.semiring.semiring_matmul', 'jit': 'pass',
             'grad': 'pass', 'vmap': 'pass', 'jit_of_grad': 'pass',
             'invariants': ['associative']},
            # not benchmarked + a failing probe -> no link, shows ✗
            {'qualname': 'nitrix.linalg.symmetric', 'jit': 'pass',
             'grad': 'TypeError: x', 'vmap': 'n/a', 'jit_of_grad': 'pass',
             'invariants': []},
        ],
    }
    html = render_site(rows, capability=capability)
    assert 'Capability matrix' in html
    assert 'nitrix.semiring.semiring_matmul' in html
    assert 'nitrix.linalg.symmetric' in html
    assert 'href="#semiring_matmul"' in html      # benchmarked op links down
    assert 'NVIDIA A10G' in html                  # capability host shown
    # a failing probe renders the ✗ glyph with the error as a title.
    assert 'title="TypeError: x"' in html


def test_no_capability_section_without_capability():
    rows = [_row('semiring_matmul', 'jax-cpu', 'nitrix-jax', 1e-3, 256)]
    assert 'Capability matrix' not in render_site(rows)


def test_history_plot_appears_only_with_multiple_runs():
    one_run = [_row('c', 'jax-cpu', 'nitrix-jax', 1e-3, 256, run_id='r1')]
    assert 'History over runs' not in render_site(one_run)
    two_runs = one_run + [
        _row('c', 'jax-cpu', 'nitrix-jax', 9e-4, 256, run_id='r2'),
    ]
    assert 'History over runs' in render_site(two_runs)
