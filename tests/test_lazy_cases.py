# -*- coding: utf-8 -*-
"""The lazy per-case registry (DESIGN §7.1).

The load-bearing property: a worker imports **only** the one case it runs, so
an isolated refs env needs only that case's deps -- not the union of all cases'
top-level imports (the over-coupling that once `env_failed` every ANTs attempt
on a stale nitrix, and forced venv-dipy to carry scikit-learn).
"""
import os
import subprocess
import sys

import pytest

from nperf import measure


def test_registry_matches_module_table():
    # The lazy registry's keys ARE the auto-derived table's keys -- every
    # cases/<name>.py (minus the `_`-prefixed helpers), nothing more.
    assert set(measure.CASES) == set(measure.CASE_MODULES)
    assert len(measure.CASES) >= 80
    # Mapping API the call sites rely on stays intact.
    assert 'ssd' in measure.CASES
    assert measure.CASES['ssd'].name == 'ssd'
    assert sorted(measure.CASES)[:1]  # iteration is cheap (keys only)


def test_name_equals_stem_invariant():
    # cases/<name>.py must export CASE.name == <name>; the auto-derived table
    # keys by file stem, so any drift would mis-key the registry.
    for name, module_path in measure.CASE_MODULES.items():
        assert module_path == 'nperf.cases.' + name
        assert measure.load_case(name).name == name


def test_load_case_memoises():
    assert measure.load_case('rigid_register') is measure.load_case(
        'rigid_register')


def test_unknown_case_raises_clearly():
    with pytest.raises(KeyError, match='unknown case'):
        measure.load_case('does_not_exist')


def test_worker_imports_only_its_case():
    # The whole point of the migration: importing `measure` + loading ONE case
    # loads only that case module (not the other 79) and pulls in none of the
    # cross-case deps -- here, sklearn (only the kernel cases top-level-import
    # it). A registration worker therefore needs no sklearn in its refs env.
    code = (
        'import sys\n'
        'import nperf.measure as m\n'
        "m.load_case('rigid_register')\n"
        "loaded = sorted(k for k in sys.modules "
        "if k.startswith('nperf.cases.') "
        "and not k.rsplit('.', 1)[1].startswith('_'))\n"
        "assert loaded == ['nperf.cases.rigid_register'], loaded\n"
        "assert 'sklearn' not in sys.modules, 'sklearn leaked in'\n"
        "print('ok')\n"
    )
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu'}
    proc = subprocess.run([sys.executable, '-c', code], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == 'ok'
