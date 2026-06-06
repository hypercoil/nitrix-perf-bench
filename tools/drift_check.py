# -*- coding: utf-8 -*-
"""Drift check: signal when a nitrix change has silently invalidated a case.

A perf-bench case encodes *assumptions* about a nitrix op -- its call
signature (which kwargs / defaults the case relies on) and what it computes on
a canonical input (which dispatch branch, what accuracy class).  When nitrix
changes underneath, those assumptions can rot while the case stays green (it
builds, the oracle still agrees within a loose tolerance), so the number it
reports quietly stops meaning what the case claims.  This tool is the tripwire.

For every registered ``Case`` it records two fingerprints of the nitrix op:

1. **signature** -- ``str(inspect.signature(op))`` resolved from
   ``Case.op_qualname`` (the *public* op, not the case's wrapper lambda).
   Catches API drift (a renamed / removed kwarg) **and default-value drift**
   (e.g. ``sosfilt``'s ``backend='scan'`` -> ``'auto'`` flip, or a changed
   ``lobpcg_tol`` default) -- both cheap and near-zero-noise.

2. **behaviour** -- a tolerance-robust digest of the op's output on the case's
   *representative* point, calling it exactly as the case does
   (``baselines['nitrix-jax']`` on ``inputs_for('jax')``).  Catches
   algorithm / dispatch / default changes that alter the output (e.g. the EDT
   going exact, a fallback firing) that a loose oracle tolerance would mask.
   The digest is **sign- and order-invariant** per leaf (L1, sum-of-squares,
   Linf, finite-count) so eigenvector sign freedom / eigenvalue-tie reordering
   do not trip it, and the stats are compared with a **relative tolerance**
   (``_BEHAVIOUR_RTOL``) so cross-process fp reassociation jitter does not
   either (only a real, gross output change trips it).

This is a **change *detector*** (nitrix vs its own committed past), not a
correctness verdict: a tripped fingerprint says "an assumption moved -- review
the case", not "nitrix is wrong".  It is the inverse of the op_matrix feed and
complements the per-case fidelity gate (which checks nitrix vs an *external*
oracle at run time; this checks nitrix vs *itself* across versions).

Run (CPU, deterministic, no GPU wedge)::

    JAX_PLATFORMS=cpu python tools/drift_check.py           # check vs manifest
    JAX_PLATFORMS=cpu python tools/drift_check.py --update  # re-bless manifest
    JAX_PLATFORMS=cpu python tools/drift_check.py --case sosfilt erode

Exit status is non-zero when any op has drifted (so CI / a pre-sprint gate
fails loudly); ``--update`` rewrites the manifest after the affected cases have
been reviewed and brought back into line.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from nperf.measure import CASES  # noqa: E402

MANIFEST = (
    Path(__file__).resolve().parents[1] / 'reports' / 'op_drift_manifest.json'
)
REPORT = Path(__file__).resolve().parents[1] / 'reports' / 'OP_DRIFT.md'

# Significant figures the stored behaviour digest is rounded to (readability of
# the manifest only -- the comparison is tolerance-based, see _BEHAVIOUR_RTOL).
_SIG = 6

# Relative tolerance for the behaviour comparison.  The digest's sum-based
# stats (l1, l2sq over millions of fp32 elements) are not bit-reproducible
# *across processes* (XLA reassociates fp reductions per compile), so an
# exact-equality compare false-alarms at the ~1e-7 level.  Comparing with
# a relative tolerance well above that jitter but far below any real algorithm
# change (a metric going exact, a fallback firing -- all >>0.1%) is robust.
_BEHAVIOUR_RTOL = 1e-3


def _sig(x: float, n: int = _SIG) -> Optional[float]:
    '''Round ``x`` to ``n`` significant figures (the digest's tolerance);
    ``None`` for non-finite so a NaN/inf appearing is itself a recorded
    change.'''
    if x is None or not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def _resolve(qualname: str):
    '''Import the public op by walking ``qualname`` (longest strict prefix is
    the module, the rest is getattr'd) -- mirrors tools/op_matrix.py so the
    signature is the one the matrix advertises.'''
    parts = qualname.split('.')
    mod = None
    rest: List[str] = []
    for i in range(len(parts) - 1, 0, -1):
        try:
            mod = importlib.import_module('.'.join(parts[:i]))
            rest = parts[i:]
            break
        except ImportError:
            continue
    if mod is None:
        raise ImportError(f'cannot import any prefix of {qualname}')
    for p in rest:
        mod = getattr(mod, p)
    return mod


def _signature(qualname: str) -> str:
    '''The op's public signature, with volatile ``repr`` memory addresses
    stripped.  Some defaults are function objects / dataclass instances
    (``null=girvan_newman_null``, ``semiring=StrictSemiring(...)``) whose
    ``repr`` embeds ``at 0x<addr>`` -- that changes every process and would
    flag spurious drift, so normalise it while keeping the meaningful content
    (the function name, the semiring ``name=``).'''
    try:
        sig = str(inspect.signature(_resolve(qualname)))
    except (ImportError, AttributeError, ValueError, TypeError) as e:
        return f'<unavailable: {type(e).__name__}>'
    return re.sub(r'0x[0-9a-fA-F]+', '0x...', sig)


def _leaf_digest(leaf: Any) -> Optional[Dict[str, Any]]:
    '''Sign/order-invariant rounded stats for one output leaf, or ``None`` for
    a non-array leaf (skipped).'''
    arr = np.asarray(leaf)
    if arr.dtype == object or arr.size == 0:
        return None
    # Complex output (analytic_signal, …): digest the magnitude so the
    # imaginary part is captured (not silently discarded) and the stats stay
    # real; dtype is recorded separately so a complex->real change still trips.
    flat = (np.abs(arr) if np.iscomplexobj(arr) else arr).reshape(-1).astype(
        np.float64)
    finite = flat[np.isfinite(flat)]
    return {
        'shape': list(arr.shape),
        'dtype': str(arr.dtype),
        'n_finite': int(finite.size),
        'n_nonfinite': int(flat.size - finite.size),
        'l1': _sig(float(np.abs(finite).sum())) if finite.size else None,
        'l2sq': _sig(float((finite * finite).sum())) if finite.size else None,
        'linf': _sig(float(np.abs(finite).max())) if finite.size else None,
    }


def _behaviour(case) -> Dict[str, Any]:
    '''Build the case at its representative point, run the nitrix-jax baseline
    the way the case calls it, and digest the output.'''
    import jax

    built = case.build(case.representative)
    provider_id, fn = built.baselines['nitrix-jax']
    out = fn(*built.inputs_for('jax'))
    out = jax.block_until_ready(out)
    leaves = [d for d in (_leaf_digest(x)
                          for x in jax.tree_util.tree_leaves(out))
              if d is not None]
    return {
        'point': {k: v for k, v in case.representative.items() if k != 'seed'},
        'leaves': leaves,
    }


def fingerprint(case) -> Dict[str, Any]:
    fp: Dict[str, Any] = {'case': case.name, 'qualname': case.op_qualname}
    if case.op_qualname:
        fp['signature'] = _signature(case.op_qualname)
    try:
        fp['behaviour'] = _behaviour(case)
    except Exception as e:  # noqa: BLE001 -- build/run failure is itself drift
        fp['behaviour'] = {'error': f'{type(e).__name__}: {str(e)[:160]}'}
    return fp


def _leaf_stat_changed(a: Any, b: Any) -> bool:
    '''One numeric stat changed beyond ``_BEHAVIOUR_RTOL`` (None-aware).'''
    if a is None or b is None:
        return a is not b
    return abs(a - b) > _BEHAVIOUR_RTOL * max(abs(a), abs(b), 1e-300)


def _leaves_changed(old_leaves: Any, new_leaves: Any) -> bool:
    '''Did the behaviour digest change beyond fp jitter?  Structural keys
    (shape, dtype, finite counts) must match exactly; the numeric stats (l1,
    l2sq, linf) are compared with a relative tolerance so cross-process fp
    reassociation does not false-alarm (see _BEHAVIOUR_RTOL).'''
    if old_leaves is None or new_leaves is None:
        return old_leaves is not new_leaves
    if len(old_leaves) != len(new_leaves):
        return True
    for o, n in zip(old_leaves, new_leaves):
        if any(o.get(k) != n.get(k)
               for k in ('shape', 'dtype', 'n_finite', 'n_nonfinite')):
            return True
        if any(_leaf_stat_changed(o.get(k), n.get(k))
               for k in ('l1', 'l2sq', 'linf')):
            return True
    return False


def _classify(old: Optional[Dict], new: Dict) -> Tuple[str, List[str]]:
    '''Compare an op's old vs new fingerprint -> (status, detail lines).'''
    if old is None:
        return 'new', ['op not in manifest']
    detail: List[str] = []
    if old.get('signature') != new.get('signature'):
        detail.append(f'signature: {old.get("signature")} -> '
                      f'{new.get("signature")}')
    ob, nb = old.get('behaviour', {}), new.get('behaviour', {})
    if 'error' in nb and 'error' not in ob:
        detail.append(f'behaviour: now errors ({nb["error"]})')
    elif 'error' in ob and 'error' not in nb:
        detail.append('behaviour: build/run recovered (was erroring)')
    elif _leaves_changed(ob.get('leaves'), nb.get('leaves')):
        detail.append('behaviour: output digest changed '
                      f'(point {nb.get("point")})')
    if not detail:
        return 'unchanged', []
    sig = any(d.startswith('signature') for d in detail)
    beh = any(d.startswith('behaviour') for d in detail)
    status = ('signature+behaviour' if sig and beh
              else 'signature' if sig else 'behaviour')
    return status, detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--update', action='store_true',
                    help='rewrite the manifest from the current state '
                         '(re-bless after reviewing flagged cases)')
    ap.add_argument('--case', nargs='+', metavar='NAME',
                    help='restrict to these cases (default: all)')
    args = ap.parse_args()

    names = args.case or sorted(CASES)
    # Keyed by CASE name, not op qualname: several cases can target one op via
    # different branches (distance_transform euclidean vs chamfer), each with
    # its own behaviour fingerprint; the qualname is recorded as a field.
    current: Dict[str, Dict[str, Any]] = {}
    for name in names:
        case = CASES[name]
        if case.op_qualname is None:
            continue  # throwaway smoke case has no public op
        current[case.name] = fingerprint(case)

    manifest = (json.loads(MANIFEST.read_text())
                if MANIFEST.exists() else {'ops': {}})
    old_ops: Dict[str, Any] = manifest.get('ops', {})

    rows: List[Tuple[str, str, str, List[str]]] = []
    drifted = 0
    for key, fp in sorted(current.items()):
        status, detail = _classify(old_ops.get(key), fp)
        rows.append((key, fp['case'], status, detail))
        if status not in ('unchanged', 'new'):
            drifted += 1

    for key, case_name, status, detail in rows:
        mark = {'unchanged': 'ok ', 'new': 'NEW'}.get(status, 'DRIFT')
        print(f'[{mark:5s}] {case_name:32s} {status}')
        for d in detail:
            print(f'           - {d}')

    if args.update:
        import jax
        merged = dict(old_ops)
        merged.update(current)  # only refresh the cases we just checked
        MANIFEST.write_text(json.dumps(
            {'_meta': {
                'generated': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                           time.gmtime()),
                'jax_version': jax.__version__,
                'sig_figs': _SIG,
                'note': 'keyed by case name (qualname is a field; >1 case may '
                        'target one op). signature + sign/order-invariant '
                        'behaviour digest; a change detector (nitrix vs its '
                        'committed past), not a correctness verdict. '
                        'tools/drift_check.py',
            }, 'ops': merged}, indent=2) + '\n')
        print(f'\nmanifest updated ({len(current)} op(s) refreshed) -> '
              f'{MANIFEST.relative_to(MANIFEST.parents[1])}')
        return

    n_new = sum(1 for *_, status, _ in rows if status == 'new')
    print(f'\n{len(rows)} op(s) checked: {len(rows) - drifted - n_new} '
          f'unchanged, {drifted} drifted, {n_new} new (not in manifest).')
    if drifted or n_new:
        print('Drift detected -- review the flagged cases, then re-bless with '
              '--update. (NEW ops: add to the manifest with --update.)')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
