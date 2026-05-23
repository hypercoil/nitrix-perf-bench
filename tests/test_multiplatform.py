# -*- coding: utf-8 -*-
"""Multi-platform fan-out, within-platform ratios, and combined rendering.

The orchestrator runs each attempt on every requested platform and rates a
baseline only against *its own platform's* reference (a GPU kernel must never
be rated against a CPU baseline).  The renderer shows the platform per row and
a per-platform host line.  Worker spawning is faked, so this is CPU-only/fast.
"""
from dataclasses import replace

from nperf import run as run_mod
from nperf.core import AttemptRecord, Status
from nperf.measure import CASES
from nperf.report import render_markdown

# numpy-matmul is dense_matmul's ratio reference; give the GPU platform a 5x
# faster reference so a *cross*-platform ratio would differ from a within one.
_BASE = {'numpy-matmul': 1.0, 'jnp-matmul': 0.5, 'jnp-einsum': 0.7,
         'jnp-bf16': 0.6}
_PLATFORM_SCALE = {'jax-cpu': 1.0, 'jax-cuda12': 0.2}


def _rec(platform: str, baseline: str, mn: float) -> AttemptRecord:
    return AttemptRecord(
        run_id='r', case='dense_matmul',
        param_point={'m': 8, 'k': 8, 'n': 8, 'seed': 0},
        baseline=baseline, platform=platform, framework='jax',
        status=Status.OK,
        metrics={'steady_time': {'min': mn, 'median': mn, 'unit': 's'}},
        provenance={
            'device': {'kind': platform, 'platform': platform},
            'measurement_isolation': 'subprocess',
        },
    )


def test_fan_out_count_and_within_platform_ratios(monkeypatch):
    def fake_spawn(spec, ctx, *, timeout):
        mn = _BASE[spec['baseline']] * _PLATFORM_SCALE[spec['platform']]
        return _rec(spec['platform'], spec['baseline'], mn)

    monkeypatch.setattr(run_mod, '_spawn_worker', fake_spawn)
    case = replace(CASES['dense_matmul'],
                   param_points=[CASES['dense_matmul'].representative])
    recs = run_mod.run_case_subprocess(
        case, platforms=['jax-cpu', 'jax-cuda12'], warmup=1, repeats=1,
        run_id='r', timeout=10, cpu_slots=1, max_parallel=2, gpu_settle_s=0.0,
    )
    assert len(recs) == 4 * 2  # 4 baselines x 2 platforms

    # Each platform's jnp-matmul is rated against *its own* numpy-matmul:
    # 0.5/1.0 on CPU and (0.5*0.2)/(1.0*0.2) on GPU -> both 0.5.  A cross-
    # platform ratio (vs the CPU reference) would make the GPU value 0.1.
    for plat in ('jax-cpu', 'jax-cuda12'):
        jnp = next(r for r in recs
                   if r.platform == plat and r.baseline == 'jnp-matmul')
        assert jnp.ratio['vs'] == 'numpy-matmul'
        assert abs(jnp.ratio['value'] - 0.5) < 1e-9


def test_render_has_platform_column_and_per_platform_host():
    rows = [_rec('jax-cpu', 'numpy-matmul', 1.0).to_json(),
            _rec('jax-cuda12', 'numpy-matmul', 0.2).to_json()]
    rows[1]['provenance']['device'] = {
        'kind': 'NVIDIA A10G', 'platform': 'gpu',
    }
    md = render_markdown(rows, rows[0]['provenance'])
    assert '| case | platform | param |' in md  # platform column
    assert '**jax-cpu**' in md and '**jax-cuda12**' in md  # per-platform host
    assert 'NVIDIA A10G' in md  # each platform's own device shown
