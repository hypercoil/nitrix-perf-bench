# -*- coding: utf-8 -*-
"""Markdown renderer (L5).

Reads L4 rows (dicts) and formats a table.  Does **no metric arithmetic** —
ratios, errors, throughput are already computed and stored in the rows
(SCHEMA_AND_LIFECYCLE §G); this only selects and formats.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _fmt_time(s: Optional[float]) -> str:
    if s is None:
        return '—'
    if s < 1e-3:
        return f'{s * 1e6:.1f} µs'
    if s < 1.0:
        return f'{s * 1e3:.2f} ms'
    return f'{s:.3f} s'


def _fmt_mem(rec: Dict[str, Any]) -> str:
    metrics = rec.get('metrics') or {}
    hbm = (metrics.get('peak_hbm') or {}).get('value')
    if hbm is not None:
        return f'{hbm:.2f} MB (hbm)'
    rss = (metrics.get('host_rss') or {}).get('value')
    return f'{rss:.0f} MB (rss)' if rss is not None else '—'


def _fmt_param(p: Dict[str, Any]) -> str:
    if {'m', 'k', 'n'} <= p.keys():
        shape = f"{p['m']}x{p['k']}x{p['n']}"
        return f"{shape} ({p['algebra']})" if 'algebra' in p else shape
    return ','.join(f'{k}={v}' for k, v in p.items() if k != 'seed')


def _fmt_fidelity(rec: Dict[str, Any]) -> str:
    fid = rec.get('fidelity')
    if not fid:
        return '—'
    tag = {'pass': '✓', 'fail': '✗', 'inconclusive': '?'}.get(
        fid.get('status'), '?'
    )
    # Headline = error as a multiple of allowed tolerance (pass <=> <= 1).
    return f"{tag} {fid.get('rel_to_tol', float('nan')):.2g}×tol"


def _fmt_ratio(rec: Dict[str, Any]) -> str:
    if rec.get('status') == 'fidelity_failed':
        return 'refused'
    ratio = rec.get('ratio')
    if not ratio:
        return '—'
    return f"{ratio['value']:.2f}x vs {ratio['vs']}"


def _mem_note(isolation: str) -> str:
    if isolation == 'subprocess':
        return (
            '- `mem` (peak_hbm / host_rss) is **per-attempt**: each attempt '
            'ran in its own process, so the high-water mark *is* this '
            'attempt\'s peak (the P1 subprocess runner; SCHEMA_AND_LIFECYCLE '
            '§B).'
        )
    return (
        '- **`mem` (peak_hbm) is a process-wide high-water mark** in this '
        'in-process run. XLA\'s `peak_bytes_in_use` does not reset between '
        'attempts, so it only ever rises: once one attempt allocates a large '
        'buffer, later rows inherit that floor. Read the *jumps* (they '
        'attribute to the attempt that caused them), not the absolute per-row '
        'value. Re-run without `--in-process` for per-attempt isolation '
        '(annex §B).'
    )


def _sched_note(sched: Dict[str, Any]) -> Optional[str]:
    slots = sched.get('cpu_slots')
    if not slots or slots <= 1:
        return None  # serial: nothing to caveat
    groups = sched.get('core_groups') or []
    return (
        f'- CPU attempts ran in **{slots} parallel slots**, each pinned to a '
        f'disjoint core group ({groups}): timings are free of cross-attempt '
        'contention, but reflect that **slot core budget**, not the whole '
        'machine — `cpu_slots=1` is the full-machine number (annex §E).'
    )


def render_markdown(
    records: List[Dict[str, Any]], prov: Dict[str, Any]
) -> str:
    dev = prov.get('device', {}) or {}
    sv = records[0].get('schema_version') if records else None
    isolation = prov.get('measurement_isolation', 'in_process')
    sched = prov.get('scheduler') or {}
    lines = [
        '# nitrix-perf-bench results',
        '',
        '> Generated from L4 result rows (SCHEMA_AND_LIFECYCLE §A); '
        f"schema_version {sv} (frozen). No values are hand-edited.",
        '',
        '## Host',
        '',
        f"- device: {dev.get('kind')} ({dev.get('platform')})",
        f"- jax: {prov.get('jax_version')} | "
        f"backend: {prov.get('jax_backend')}",
        f"- precision: {prov.get('precision_policy')} | "
        f"x64: {prov.get('jax_enable_x64')} | "
        f"preallocate: {prov.get('xla_preallocate')} | "
        f"compile_cache: {prov.get('compile_cache')} | "
        f"isolation: {isolation}",
        f"- scheduler: cpu_slots={sched.get('cpu_slots')} | "
        f"max_parallel={sched.get('max_parallel')} | "
        f"gpu_settle_s={sched.get('gpu_settle_s')}" if sched else None,
        f"- nitrix: {(prov.get('nitrix') or {}).get('sha')} | "
        f"bench: {(prov.get('bench') or {}).get('sha')}",
        f"- {prov.get('os')} | python {prov.get('python')} | "
        f"{prov.get('timestamp')}",
        '',
        '## Measurements',
        '',
        '`steady` = post-warm-up min / median; `compile` = cold first call; '
        '`fidelity` = worst error as a multiple of the allowed tolerance vs '
        'the fp64 oracle (✓ pass ⟺ ≤ 1×tol / ✗ fail). A `fidelity_failed` row '
        'keeps its measurements but its ratio is `refused`.',
        '',
        '| case | param | baseline | status | steady (min/med) | compile | '
        'mem | fidelity | ratio |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for r in records:
        metrics = r.get('metrics') or {}
        steady = metrics.get('steady_time') or {}
        comp = metrics.get('compile_time') or {}
        steady_cell = (
            f"{_fmt_time(steady.get('min'))} / "
            f"{_fmt_time(steady.get('median'))}"
            if steady else '—'
        )
        lines.append(
            '| {case} | {param} | `{baseline}` | {status} | {steady} | '
            '{compile} | {mem} | {fid} | {ratio} |'.format(
                case=r.get('case'),
                param=_fmt_param(r.get('param_point', {})),
                baseline=r.get('baseline'),
                status=r.get('status'),
                steady=steady_cell,
                compile=_fmt_time(comp.get('value')),
                mem=_fmt_mem(r),
                fid=_fmt_fidelity(r),
                ratio=_fmt_ratio(r),
            )
        )
    lines += [
        '',
        '## Notes',
        '',
        '- `steady` is the post-warm-up min / median of the timed loop; '
        '`ratio` is on `min` vs the reference baseline, computed and stored '
        'in L1 (this renderer does no metric arithmetic).',
        '- `compile` is the **cold** first-call cost (`jax.clear_caches()` '
        'per attempt; persistent cache disabled) — what a user pays once, '
        'not a steady-state number.',
        '- `fidelity` is `rel_to_tol`: the worst error as a multiple of the '
        'allowed tolerance vs the fp64 oracle (✓ ⟺ ≤ 1×tol). It is '
        'tolerance-relative on purpose — a bare relative error is meaningless '
        'for zero-centred outputs (SCHEMA_AND_LIFECYCLE §C).',
        _mem_note(isolation),
        _sched_note(sched),
        '',
    ]
    return '\n'.join(x for x in lines if x is not None) + '\n'
