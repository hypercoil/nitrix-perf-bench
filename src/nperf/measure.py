# -*- coding: utf-8 -*-
"""Shared L1/L3 measurement core — used by *both* the in-process driver
(`run.py`) and the single-attempt subprocess worker (`worker.py`).

Keeping the per-attempt measurement here (rather than in either entrypoint)
means the worker and the in-process path measure *identically* — the only
difference is process isolation, which is what makes per-attempt memory honest
(see `worker.py` / SCHEMA_AND_LIFECYCLE §B).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import numpy as np

from .cases import (
    BuiltPoint,
    Case,
    analytic_signal,
    bilateral_gaussian,
    coaffiliation,
    corr,
    cosine_kernel,
    cov,
    degree_vector,
    diffusion_embedding,
    dilate,
    distance_transform,
    ell_edge_aggregate,
    envelope,
    erode,
    flame_two_level,
    gaussian,
    gaussian_kernel,
    girvan_newman_null,
    hilbert_transform,
    histogram_match,
    laplacian,
    laplacian_eigenmap,
    linear_distance,
    linear_kernel,
    lomb_scargle_interpolate,
    lomb_scargle_periodogram,
    median_filter,
    modularity_matrix,
    n4_bias_field_correction,
    partialcorr,
    partialcov,
    polynomial_detrend,
    polynomial_kernel,
    precision,
    rbf_kernel,
    relaxed_modularity,
    reml_fit,
    resample,
    residualise,
    semiring_matmul,
    sigmoid_kernel,
    sosfilt,
    sosfiltfilt,
    spatial_transform,
    symexp,
    symlog,
    sympower,
    symsqrt,
    tangent_project_spd,
    throwaway,
    tsconv,
)
from .core import (
    METRICS,
    AttemptRecord,
    Status,
    bench_call,
    compare,
    host_rss_mb,
    peak_hbm_mb,
)
from .core.sync import SYNC
from .providers import framework_of


def _host_f64(out: Any, framework: str) -> np.ndarray:
    '''Bring a baseline output to host at high precision for the fidelity
    compare: fp64, or **complex128** for a complex output (e.g.
    ``analytic_signal``).

    torch tensors (possibly on cuda, possibly autograd-tracked) need an
    explicit ``detach().cpu().numpy()`` -- ``np.asarray`` rejects a cuda
    tensor.  cupy needs ``cp.asnumpy``.  jax arrays / numpy convert directly.
    torch / cupy are imported only when handed such an output, so the base env
    never needs them here.'''
    if framework == 'torch':
        arr = out.detach().cpu().numpy()
    elif framework == 'cupy':
        import cupy as cp

        arr = cp.asnumpy(out)
    else:
        arr = np.asarray(out)
    return arr.astype(np.complex128 if np.iscomplexobj(arr) else np.float64)


def _validate_case(case: Case) -> Case:
    '''Reject a case naming a metric outside the registry (typo-proofing).'''
    unknown = [m for m in case.metrics if m not in METRICS]
    if unknown:
        raise ValueError(
            f'case {case.name!r}: unknown metric(s) {unknown}; '
            f'register them in core/metrics.py (known: {sorted(METRICS)}).'
        )
    return case


# The case registry (L2).  Lives here so both entrypoints share one source;
# each case is validated against the metric registry on registration.
CASES: Dict[str, Case] = {
    c.name: c
    for c in (_validate_case(throwaway.CASE),
              _validate_case(semiring_matmul.CASE),
              _validate_case(sosfilt.CASE),
              _validate_case(sosfiltfilt.CASE),
              _validate_case(ell_edge_aggregate.CASE),
              _validate_case(cov.CASE),
              _validate_case(corr.CASE),
              _validate_case(residualise.CASE),
              _validate_case(resample.CASE),
              _validate_case(gaussian.CASE),
              _validate_case(bilateral_gaussian.CASE),
              _validate_case(erode.CASE),
              _validate_case(dilate.CASE),
              _validate_case(distance_transform.CASE),
              _validate_case(spatial_transform.CASE),
              _validate_case(median_filter.CASE),
              _validate_case(histogram_match.CASE),
              _validate_case(n4_bias_field_correction.CASE),
              _validate_case(laplacian.CASE),
              _validate_case(laplacian_eigenmap.CASE),
              _validate_case(diffusion_embedding.CASE),
              _validate_case(modularity_matrix.CASE),
              _validate_case(degree_vector.CASE),
              _validate_case(girvan_newman_null.CASE),
              _validate_case(coaffiliation.CASE),
              _validate_case(relaxed_modularity.CASE),
              _validate_case(symexp.CASE),
              _validate_case(symlog.CASE),
              _validate_case(symsqrt.CASE),
              _validate_case(sympower.CASE),
              _validate_case(tangent_project_spd.CASE),
              _validate_case(analytic_signal.CASE),
              _validate_case(hilbert_transform.CASE),
              _validate_case(envelope.CASE),
              _validate_case(rbf_kernel.CASE),
              _validate_case(linear_kernel.CASE),
              _validate_case(linear_distance.CASE),
              _validate_case(gaussian_kernel.CASE),
              _validate_case(cosine_kernel.CASE),
              _validate_case(polynomial_kernel.CASE),
              _validate_case(sigmoid_kernel.CASE),
              _validate_case(polynomial_detrend.CASE),
              _validate_case(tsconv.CASE),
              _validate_case(lomb_scargle_periodogram.CASE),
              _validate_case(precision.CASE),
              _validate_case(partialcov.CASE),
              _validate_case(partialcorr.CASE),
              _validate_case(reml_fit.CASE),
              _validate_case(flame_two_level.CASE),
              _validate_case(lomb_scargle_interpolate.CASE))
}


def platform_from(prov: Dict[str, Any]) -> str:
    '''Map the captured backend to an env-group id (L6).'''
    backend = (prov.get('device') or {}).get('platform')
    return {'gpu': 'jax-cuda12', 'cpu': 'jax-cpu'}.get(
        backend, f'jax-{backend}'
    )


def classify_message(msg: str) -> Tuple[Status, Dict[str, Any]]:
    '''Classify a failure *message* into a status + failure_detail.

    Shared by the in-process exception path and the orchestrator's
    worker-death path (which only has the dead process's stderr text).'''
    low = msg.lower()
    if 'resource_exhausted' in low or 'out of memory' in low:
        return Status.OOM, {'message': msg}
    # A provider's framework package isn't importable in the worker env (e.g.
    # the torch refs env was never built / NPERF_PYTHON_TORCH is unset, so the
    # worker fell back to the jax-only base env).  The *env* is wrong, not the
    # op -- a clean env_failed, so the default CPU run degrades gracefully
    # (torch-dense records env_failed; the jax baselines still run).
    if 'no module named' in low or 'modulenotfounderror' in low:
        return Status.ENV_FAILED, {
            'phase': 'import', 'reason': 'provider_env_missing',
            'message': msg,
        }
    # Requested backend / device simply isn't present on this host (e.g. the
    # pallas-cuda kernel on a CPU box).  The env imported fine and nothing
    # failed to compile -- the hardware is absent -- so this is a recorded
    # *skip*, not a compile_error.  Message-heuristic, same style as OOM above.
    if 'visible' in low and ('gpu' in low or 'device' in low):
        return Status.SKIPPED, {
            'reason': 'backend_unavailable', 'message': msg,
        }
    # A genuine GPU cuSolver failure (e.g. jax's GPU eigh/solvers on this CUDA
    # stack -- a *jaxlib* bug; cupy's eigh works on the identical bundled
    # wheels).  PRECISE on purpose: only a real cuSolver signature trips this,
    # so the reason name stays accurate and no unrelated GPU failure is
    # mislabelled (jax-ml/jax #29042; the gpu-eigh-blocker note).
    if 'cusolver' in low or 'gpusolverdncreate' in low:
        return Status.SKIPPED, {
            'reason': 'gpu_solver_unavailable', 'message': msg,
        }
    # A requested device backend is absent in this single-backend worker --
    # e.g. nitrix's safe_eigh CPU fallback finding no CPU device on a gpu-only
    # worker ("Unknown backend cpu").  Accurate + generic; NOT solver-specific
    # (the underlying cause -- here the cuSolver bug -- is documented per case,
    # not asserted from this string).
    if 'unknown backend' in low:
        return Status.SKIPPED, {
            'reason': 'backend_unavailable', 'message': msg,
        }
    return Status.COMPILE_ERROR, {'message': msg}


def classify_exception(exc: Exception) -> Tuple[Status, Dict[str, Any]]:
    '''Classify a caught attempt exception into a status + failure_detail.'''
    return classify_message(f'{type(exc).__name__}: {exc}')


def measure_attempt(
    case: Case,
    param: Dict[str, Any],
    built: BuiltPoint,
    baseline_name: str,
    *,
    platform: str,
    run_id: str,
    prov: Dict[str, Any],
    warmup: int,
    repeats: int,
) -> AttemptRecord:
    '''Measure one ``(case, param, baseline)`` against the point's oracle.

    The per-attempt lifecycle the schema depends on: ``jax.clear_caches()``
    first (cold compile, annex §D), time the warm steady state, compare to the
    shared fp64 oracle, and on the fidelity gate *refuse the ratio but record
    the absolutes* (DESIGN §1).  When the case has **no cross-impl oracle**
    (``built.fp64_reference is None`` -- baselines compute the same task but
    not bit-identical results), the row is still OK with an *inconclusive*
    fidelity block (the perf ratio stays a fair task-level comparison).  Any
    exception becomes a classified status row — failure is data, never fatal.
    '''
    # The case-local baseline name maps to (provider_id, run_fn); the provider
    # registry says which framework runs it (sync hook / jit).
    provider_id, fn = built.baselines[baseline_name]
    framework = framework_of(provider_id)
    base = dict(
        run_id=run_id, case=case.name, param_point=param,
        baseline=baseline_name, platform=platform, framework=framework,
        provenance=prov,
    )
    try:
        jax.clear_caches()  # cold compile per attempt (annex §D)
        args = built.inputs_for(framework)
        run_fn = jax.jit(fn) if framework == 'jax' else fn
        sync = SYNC[framework]
        compile_s, dist = bench_call(
            run_fn, args, warmup=warmup, repeats=repeats, sync=sync,
        )
        out = run_fn(*args)
        sync(out)
        out_host = _host_f64(out, framework)
        # Units come from the metric registry (single source of truth).
        metrics = {
            'steady_time': {**dist.summary(),
                            'unit': METRICS['steady_time'].unit},
            'compile_time': {'value': compile_s,
                             'unit': METRICS['compile_time'].unit,
                             'cache': 'cold'},
            'peak_hbm': {'value': peak_hbm_mb(framework),
                         'unit': METRICS['peak_hbm'].unit},
            'host_rss': {'value': host_rss_mb(),
                         'unit': METRICS['host_rss'].unit},
            'throughput': {
                'value': float(out_host.size) / dist.min,
                'unit': METRICS['throughput'].unit,
            },
        }
        # No cross-implementation oracle: the case's baselines compute the same
        # *task* but not bit-identical results (e.g. a different boundary
        # convention), so there is no fp64 ground truth both should match.  The
        # measurement is still OK and the perf ratio is still a fair task-level
        # comparison; only the numerical check is N/A -- recorded as an
        # inconclusive fidelity block (SCHEMA §C/§F), not a gate failure.
        if built.fp64_reference is None:
            fid = {'status': 'inconclusive',
                   'reason': built.fidelity_note or 'no_cross_impl_oracle'}
            return AttemptRecord(
                **base, status=Status.OK, metrics=metrics, fidelity=fid,
            )
        fid = compare(
            out_host, built.fp64_reference, rtol=case.rtol, atol=case.atol,
        )
        if fid['status'] == 'pass':
            return AttemptRecord(
                **base, status=Status.OK, metrics=metrics, fidelity=fid,
            )
        # Refuse the ratio, but keep the absolutes + the failing record.
        return AttemptRecord(
            **base, status=Status.FIDELITY_FAILED, metrics=metrics,
            fidelity=fid, failure_detail={'fidelity': fid},
        )
    except Exception as exc:  # noqa: BLE001 -- failure is data; classified.
        status, detail = classify_exception(exc)
        return AttemptRecord(**base, status=status, failure_detail=detail)


def attach_ratios(
    attempts: List[AttemptRecord], ratio_reference: str
) -> None:
    '''Fill ``ratio`` (vs the reference baseline, on ``min``) for the OK rows
    of one param point.  Stored in L1 — not recomputed in a renderer (§G).'''
    ref = next(
        (a for a in attempts
         if a.baseline == ratio_reference and a.status == Status.OK), None,
    )
    if ref is None or not ref.metrics:
        return
    ref_min = ref.metrics['steady_time']['min']
    for a in attempts:
        if a.status == Status.OK and a.metrics:
            a.ratio = {
                'vs': ratio_reference, 'metric': 'min',
                'value': a.metrics['steady_time']['min'] / ref_min,
            }
