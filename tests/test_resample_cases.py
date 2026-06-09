# -*- coding: utf-8 -*-
"""Tier-2 resample case -- the interpolation-kernel branches (Phase 3).

Image resize (align_corners=True) across the dispatcher's kernels. The spline
kernels (Linear / NearestNeighbour / CubicBSpline) have an exact scipy
``map_coordinates`` analogue (order 1 / 0 / 3, cubic with mirror prefilter), so
they gate against a clean fp64 oracle; Lanczos has no analogue (perf-only, no
oracle). The ANTs + cupy refs need other envs (checked in the matrix run).
"""
import numpy as np

from nperf.cases import resample
from nperf.core.fidelity import compare
from nperf.providers import framework_of, requires_of


def _p(kernel):
    return {'shape': [48, 48, 48], 'out': [64, 64, 64], 'kernel': kernel,
            'seed': 0}


def test_linear_carries_ants_and_gpu_refs():
    built = resample._build({'shape': [48, 48], 'out': [96, 96],
                             'kernel': 'linear', 'seed': 0})
    assert {'nitrix-jax', 'ants.resample_image',
            'scipy.ndimage.map_coordinates',
            'cupyx.scipy.ndimage.map_coordinates'} == set(built.baselines)
    assert built.ratio_reference == 'nitrix-jax'
    assert requires_of(built.baselines['ants.resample_image'][0]) == 'cpu'
    assert requires_of(
        built.baselines['cupyx.scipy.ndimage.map_coordinates'][0]) == 'gpu'


def test_spline_kernels_gate_against_scipy_oracle():
    # Linear / Nearest / Cubic: nitrix + scipy match the fp64 oracle.
    for kernel in ('linear', 'nearest', 'cubic'):
        built = resample._build(_p(kernel))
        assert built.fp64_reference is not None
        for name, (pid, fn) in built.baselines.items():
            if framework_of(pid) not in ('jax', 'numpy'):
                continue  # ants / cupy: other envs
            out = np.asarray(fn(*built.inputs_for(framework_of(pid))))
            fid = compare(out, built.fp64_reference,
                          rtol=resample.CASE.rtol, atol=resample.CASE.atol)
            assert fid['status'] == 'pass', (
                f'{kernel}/{name}: rel_to_tol={fid["rel_to_tol"]:.3g}')


def test_lanczos_is_perf_only_no_oracle():
    # No map_coordinates analogue + nitrix is the ANTs algorithm class (not
    # bit-exact ITK parity) -> no cross-impl oracle; nitrix-only perf point.
    built = resample._build(_p('lanczos'))
    assert built.fp64_reference is None and built.fidelity_note
    assert set(built.baselines) == {'nitrix-jax'}


def test_case_covers_the_kernel_branches():
    kernels = {p.get('kernel') for p in resample.CASE.param_points}
    assert {'linear', 'nearest', 'cubic', 'lanczos'} <= kernels


def test_op_qualname():
    assert resample.CASE.op_qualname == 'nitrix.geometry.resample'
