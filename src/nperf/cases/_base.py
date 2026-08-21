# -*- coding: utf-8 -*-
"""Case + BuiltPoint protocol (L2).

A ``Case`` declares an op under test: the param points it spans, its
*representative* point (the coverage-tier anchor, DESIGN §1/§4), whether its
outputs are *element-wise independent* (gates fp64 subsampling, annex §C), the
metrics it supports, and a ``build(param)`` that materialises one
``BuiltPoint``.

A ``BuiltPoint`` carries the competing implementations and the shared oracle
for one param point: per-framework on-device inputs, the baselines
(``case-local name -> (provider_id, run_fn)``; the ``provider_id`` keys the
cross-case provider registry in ``nperf.providers`` for framework + env
isolation), the fp64 ground-truth output (computed once, shared across
baselines), and which baseline ratios are taken against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


def to_cupy(*arrays: Any) -> Tuple[Any, ...]:
    '''Host arrays -> on-device CuPy arrays, synchronised so the H2D transfer
    sits *outside* the timed region (mirrors the ``jax.block_until_ready`` on
    the jax inputs in a case ``build``).  ``cupy`` is imported lazily, so only
    the cupy worker (the refs-cupy env) ever needs it -- a jax/numpy worker
    that calls a case's ``inputs_for`` for its own framework never imports it.
    Used by the audit cases' GPU reference (Phase B).'''
    import cupy as cp

    out = tuple(cp.asarray(a) for a in arrays)
    cp.cuda.runtime.deviceSynchronize()
    return out


@dataclass(frozen=True)
class SlowBaseline:
    '''A baseline whose *measurement* is pathologically expensive (not its
    steady state) -- e.g. a cold compile of minutes -- so a dev-cycle run may
    skip it via ``--skip-slow`` (`run.py`).

    ``reason`` is the audit trail (why + the measured cost, like a provider's
    ``pixi_reason``): slowness is *evidence-based and hardware-dependent*, so
    record the number and the device it was seen on, and re-evaluate on
    re-bench rather than treating the entry as eternal.

    Known limitation (deferred, no action yet): slowness is currently
    **platform-flat** -- ``--skip-slow`` drops the baseline on *every*
    platform, even when it is slow on only one.  Measured counter-example:
    ``naive-dense`` costs ~432s to compile on the L4 GPU but only ~0.3s on CPU,
    so a GPU-driven skip needlessly drops the cheap, useful CPU measurement.  A
    ``platforms: Optional[Tuple[str, ...]]`` field to scope the skip is the
    refinement; deferred because dev cycles run both platforms together and the
    GPU compile is the long pole, so the flat skip is acceptable for now.'''
    baseline: str
    reason: str


@dataclass(frozen=True)
class ApproxBaseline:
    '''A baseline that computes the case's task only **approximately** -- a
    fidelity/speed tradeoff, not a bug.  Its fidelity is *reported, not gated*:
    the row stays ``OK`` and still earns a ratio, but its fidelity block is
    marked ``'approximate'`` carrying the measured ``rel_to_tol`` -- the
    approximation magnitude *is* the signal, read against the speed (the
    quantised-inference case: practitioners knowingly trade accuracy for
    throughput).  This is distinct from a true reference, which must pass the
    case's tight gate, and from a fidelity *failure*, which refuses the ratio.

    ``reason`` is the audit trail -- what the approximation is, its measured
    magnitude, and the device it was seen on -- like ``SlowBaseline``:
    evidence-based, re-checked on re-bench rather than treated as eternal.
    '''
    baseline: str
    reason: str


@dataclass(frozen=True)
class CostLaw:
    '''Machine-readable companion to the prose ``complexity``: the op's
    *theoretical* asymptotic cost exponent in a named scale axis, so the
    extrapolation tool (``tools/extrapolate_report.py``) can (a) check the
    empirically-fitted log-log exponent against theory, (b) bracket the
    brain-scale projection between the empirical fit and a theory-anchored
    slope, and (c) flag when the two diverge (a small-n constant-factor regime
    not yet asymptotic, or a genuine surprise).  ``complexity`` stays the human
    narrative; this is its structured form.

    ``axis`` is the param-point KEY the curve fits on (the scale driver -- e.g.
    ``'V'`` for a voxel batch, ``'q'`` for the random-effect level count);
    ``time_exp`` / ``hbm_exp`` are the expected exponents of ``steady_time`` /
    ``peak_hbm`` in that axis (HBM is usually elements-linear).  ``regime``
    records the asymptotic regime the law assumes (e.g.
    ``'many-tier q>64 structured'``), since a different modelling path through
    the same op can have a different exponent.'''
    axis: str
    time_exp: float
    hbm_exp: float = 1.0
    regime: str = ''


@dataclass(frozen=True)
class ScalePath:
    '''One *modelling path* of an op -- a cell of its config space (e.g. a GLMM
    ``family`` x ``structure`` x ``method`` x level-count tier) -- measured
    across a dense range of small, fast ``grid`` scales along ``cost.axis``, so
    its brain-scale cost can be *extrapolated* (the empirical fit combined with
    ``cost``) rather than measured at the prohibitive full size.  The headline
    brain-scale point itself is one ``Case.large_param_points`` anchor carrying
    the same ``label`` -- the extrapolation is validated against it.

    ``params`` are the path-fixing keys a case ``_build`` branches on (the
    existing param-dict-branching idiom, cf. ``reml_fit``'s ``data`` branch);
    ``scaling_sweep`` crosses them with ``grid`` into the ``param_points``.
    ``challenging`` marks a numerically hard path (the cells v3 hardened
    against divergence / overflow / indefinite Hessians) so the report can
    surface that performance is retained there too.'''
    label: str
    params: Dict[str, Any]
    cost: CostLaw
    grid: Tuple[Any, ...]
    challenging: bool = False


def scaling_sweep(
    paths: List[ScalePath], base: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    '''Cross each modelling ``path`` with its small-fast ``grid`` into the flat
    ``param_points`` list the runner sweeps.  Each emitted dict is
    ``base | path.params | {path.cost.axis: v, 'path': path.label}`` -- the
    ``'path'`` label lets the case ``_build`` branch on the path and lets
    ``extrapolate_report`` group rows by path and fit over ``path.cost.axis``.
    A ``challenging`` path also stamps ``'challenging': True`` so the report
    can flag it from any of its rows.'''
    base = base or {}
    out: List[Dict[str, Any]] = []
    for p in paths:
        for v in p.grid:
            pt: Dict[str, Any] = {
                **base, **p.params, p.cost.axis: v, 'path': p.label}
            if p.challenging:
                pt['challenging'] = True
            out.append(pt)
    return out


@dataclass
class BuiltPoint:
    # case-local baseline name -> (provider_id, run_fn(*args) -> output);
    # provider_id keys nperf.providers (framework + env isolation).
    baselines: Dict[str, Tuple[str, Callable[..., Any]]]
    # framework -> on-device args (host->device transfer excluded from timing)
    inputs_for: Callable[[str], Tuple[Any, ...]]
    # ground-truth output, host fp64, computed once for this param point.
    # ``None`` means there is **no cross-implementation oracle** (the baselines
    # compute the same task but not bit-identical results -- e.g. a different
    # boundary convention); the attempt is then OK with an *inconclusive*
    # fidelity block instead of a gate comparison (see measure_attempt).
    fp64_reference: Any
    # baseline name that ratios are computed against
    ratio_reference: str
    # when fp64_reference is None: the human reason the comparison is N/A
    # (recorded as fidelity.reason): why the baselines legitimately differ.
    fidelity_note: Optional[str] = None
    # OPTIONAL registration-recovery scorer (REGISTRATION_RECOVERY): a
    # planted-warp point with a known ground truth carries a closure
    # ``recovery(name, out_host) -> {recovery_ncc, recovery_tre, ...}``
    # (built in ``_build`` capturing the GT field / fixed image / mask).  When
    # present, ``measure_attempt`` scores each baseline's output against the
    # planted truth and injects the scalars into ``metrics`` -- so recovery
    # quality is reported *beside* speed.  ``None`` -> speed only (as today).
    # The closure dispatches on ``baseline_name`` because nitrix baselines
    # return the transform/field while community refs return the warped image.
    recovery: Optional[Callable[[str, Any], Dict[str, float]]] = None


@dataclass
class Case:
    name: str
    output_independent: bool
    metrics: List[str]
    param_points: List[Dict[str, Any]]
    representative: Dict[str, Any]
    build: Callable[[Dict[str, Any]], BuiltPoint]
    # per-case fidelity tolerance (np.allclose semantics)
    rtol: float = 1e-3
    atol: float = 1e-4
    # baselines whose *measurement* is pathologically slow (cold compile, …);
    # ``--skip-slow`` drops these for fast dev cycles (recorded as skipped
    # rows, and the run is stamped ``coverage_mode = fast`` so the op_matrix
    # feed / gate refuse to treat it as authoritative -- run a full sweep at
    # sprint end).
    slow_baselines: Tuple[SlowBaseline, ...] = field(default_factory=tuple)
    # baselines that compute the task only *approximately* (a fidelity/speed
    # tradeoff, e.g. a 4SED distance transform or a quantised kernel): their
    # fidelity is reported, not gated -- the row stays OK with an
    # ``'approximate'`` fidelity block and still earns a ratio, so the
    # accuracy-vs-speed tradeoff is visible, not dropped (ApproxBaseline).
    approximate_baselines: Tuple[ApproxBaseline, ...] = field(
        default_factory=tuple)
    # the public nitrix op this case measures (its op_matrix qualname); the
    # single home for the case -> op mapping the op_matrix feed and the
    # decision-input bundle both read.  None for the throwaway smoke case.
    op_qualname: Optional[str] = None
    # benchmarking-policy tier (COVERAGE_V2_PLAN): ``'marquee'`` ops are the
    # headline functions used on real images -- the coverage matrix *requires*
    # them to be tested on real brain data against a community gold standard
    # (real_planted+ realism + a domain ref on real data); ``'standard'`` ops
    # are not held to that bar.  Lives here (policy), not in op_matrix.json
    # (capability).
    tier: str = 'standard'
    # **brain-scale** param points (the size tier): the realistic large /
    # batched sizes a perf win must still hold at -- the defence against
    # *scale-gaming* (notching a win at a small benched size while a worse
    # asymptotic / memory growth loses, or OOMs, before the scale
    # practitioners run; cf. ApproxBaseline for accuracy-gaming).  Run by
    # default *in addition to* ``param_points``; ``--skip-large`` drops them
    # for fast dev cycles and stamps the run ``coverage_mode = fast``
    # (non-authoritative, like ``--skip-slow``).  Kept distinct from
    # ``param_points`` so the ``representative`` (drift / dev anchor) stays
    # small while the scaling curve + crossover are measured here (read by
    # ``tools/scaling_report.py``).
    large_param_points: Tuple[Dict[str, Any], ...] = field(
        default_factory=tuple)
    # the op's asymptotic cost *law* (a derived, warranted statement -- time
    # and HBM, nitrix vs the reference), so a crossover is *predictable* from
    # the algorithm, not just observed at whichever sizes we happened to pick.
    # Surfaced by ``tools/scaling_report.py`` beside the measured curve.
    complexity: Optional[str] = None
    # machine-readable cost law(s) -- the structured companion to
    # ``complexity`` that ``tools/extrapolate_report.py`` fits the empirical
    # scaling curve against (theory-vs-empirical exponent + a brain-scale
    # projection bracket validated against the ``large_param_points`` anchor).
    # A SINGLE-path op sets ``cost_law`` (its whole sweep is one law); a
    # MULTI-path op -- a config space, e.g. GLMM family x structure x method x
    # level-count tier -- declares ``scale_paths`` (one ``ScalePath`` per path,
    # each with its own axis, exponent, and small-fast grid; ``param_points``
    # are built from them via ``scaling_sweep``).  Both optional, so existing
    # cases retrofit lazily.
    cost_law: Optional[CostLaw] = None
    scale_paths: Tuple[ScalePath, ...] = field(default_factory=tuple)
