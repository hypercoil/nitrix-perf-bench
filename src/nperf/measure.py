# -*- coding: utf-8 -*-
"""Shared L1/L3 measurement core — used by *both* the in-process driver
(`run.py`) and the single-attempt subprocess worker (`worker.py`).

Keeping the per-attempt measurement here (rather than in either entrypoint)
means the worker and the in-process path measure *identically* — the only
difference is process isolation, which is what makes per-attempt memory honest
(see `worker.py` / SCHEMA_AND_LIFECYCLE §B).
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import numpy as np

# Only the base types -- NOT the case modules (DESIGN §7.1: the case registry
# is lazy, so importing `measure` no longer drags in all 80 cases + their
# top-level deps; a worker imports just the one case it runs via `load_case`).
from .cases import BuiltPoint, Case
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


# The case registry (L2), lazy (DESIGN §7.1).  A case is `cases/<name>.py`
# exporting `CASE` with `CASE.name == <name>` (the file-stem == case-name
# invariant, asserted by `load_case` + tests/test_lazy_cases.py); `_`-prefixed
# modules are shared helpers, not cases.  The name->module-path table is built
# by *listing* the directory -- it imports nothing -- so importing `measure`
# (and thus a worker) costs only the one case it later loads, not all 80 +
# their top-level deps.  Adding a case is just dropping a file: no edit here.
_CASES_DIR = Path(__file__).resolve().parent / 'cases'
CASE_MODULES: Dict[str, str] = {
    p.stem: '%s.cases.%s' % (__package__, p.stem)
    for p in _CASES_DIR.glob('*.py')
    if not p.stem.startswith('_')
}

_CASE_CACHE: Dict[str, Case] = {}


def load_case(name: str) -> Case:
    '''Import + validate the single case ``name`` and return its ``CASE``.

    **The only place a case module is imported** -- so a worker pulls in just
    the case it runs (DESIGN §7.1), not the whole registry.  Asserts the
    file-stem == case-name invariant, runs ``_validate_case`` (metric
    registry), and memoises so repeated loads in one process are free.'''
    cached = _CASE_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        module_path = CASE_MODULES[name]
    except KeyError:
        raise KeyError(
            f'unknown case {name!r}; known: {sorted(CASE_MODULES)} '
            '(a case is cases/<name>.py exporting CASE with name == <stem>).'
        ) from None
    case = importlib.import_module(module_path).CASE
    if case.name != name:
        raise ValueError(
            f'case file cases/{name}.py exports CASE.name={case.name!r}; the '
            'file stem and case name must match (DESIGN §7.1 invariant) -- '
            f'rename the file to cases/{case.name}.py or fix CASE.name.'
        )
    _CASE_CACHE[name] = _validate_case(case)
    return _CASE_CACHE[name]


class _CaseRegistry(Mapping):
    '''Lazy, dict-like view of the case registry (DESIGN §7.1).

    Keys / iteration / ``len`` / ``in`` are cheap (the ``CASE_MODULES`` table,
    no imports); ``__getitem__`` imports one case via ``load_case``; the
    inherited ``.values()`` / ``.items()`` import every case on demand (used
    only by the base-env coverage / scaling / op-matrix tools that genuinely
    want the whole registry).  Drop-in for the old eager ``{name: CASE}`` dict
    so all call sites -- ``sorted(CASES)``, ``CASES[name]``, ``CASES.values()``
    -- keep working unchanged.'''

    def __getitem__(self, name: str) -> Case:
        return load_case(name)

    def __iter__(self):
        return iter(CASE_MODULES)

    def __len__(self) -> int:
        return len(CASE_MODULES)

    def __contains__(self, name: object) -> bool:
        return name in CASE_MODULES


CASES: Mapping = _CaseRegistry()


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
        # Declared-approximate baseline (a fidelity/speed tradeoff, e.g. a 4SED
        # distance transform or a quantised kernel): report the gap, keep the
        # row OK and let it earn a ratio -- the approximation magnitude set
        # against the speed *is* the signal (ApproxBaseline).
        if baseline_name in {a.baseline for a in case.approximate_baselines}:
            return AttemptRecord(
                **base, status=Status.OK, metrics=metrics,
                fidelity={**fid, 'status': 'approximate'},
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
