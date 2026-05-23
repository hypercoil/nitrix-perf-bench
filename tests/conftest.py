# -*- coding: utf-8 -*-
"""Test-suite fixtures / environment.

The unit suite is **CPU-only**: it checks the schema, the fidelity arithmetic,
and that the cases build correct outputs — none of which needs a GPU.  Real GPU
*performance* runs go through the runner (`nperf` on a CUDA host), not pytest.
We force ``JAX_PLATFORMS=cpu`` here (the host may export ``cuda,cpu``, which a
CPU-only jax install cannot initialise) before jax is imported anywhere.
"""
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
