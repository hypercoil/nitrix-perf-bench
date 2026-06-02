#!/usr/bin/env bash
# Build a P2 cross-framework "refs" worker env (torch + torch_geometric + jax +
# nitrix), in a CPU or a CUDA variant.
#
# Why a separate env: nitrix's own env is jax-only and must NEVER import torch
# (DESIGN §7); the torch / PyG reference baselines therefore run in their *own*
# subprocess under *this* interpreter, selected by the runner via
# `NPERF_PYTHON_TORCH` (cpu) / `NPERF_PYTHON_TORCH_JAX_CUDA12` (gpu).  It is a
# superset env (torch + torch_geometric for the baselines, jax + nitrix for the
# shared fp64 oracle / provenance every worker rebuilds, and the core-dep
# reference libs scipy + scikit-learn -- every worker imports measure.py, which
# imports every case module, so their top-level refs must resolve here too).
# torch_geometric >=2.3
# message-passes on torch-native scatter_reduce, so it installs pure-Python --
# no compiled torch-scatter/torch-sparse, no pixi.
#
# The CUDA variant installs jax[cuda12] *and* a CUDA torch in one env: jax 0.10
# pulls nvidia-*-cu13 and torch pulls nvidia-*-cu12 -- different CUDA major
# families / separate packages, so they coexist (each dlopens its own bundled
# libs) and both see the GPU.  This keeps provenance honest (jax reports the
# real device) with no runner changes.
#
# This env is a host artifact, NOT committed.  It lives off the tiny root
# overlay (torch+cuda is several GB).  Treat the target dir as ephemeral (it can
# vanish across sessions) -- this script is the reproducible recipe; re-run it.
#
# Usage:
#   tools/setup_refs_env.sh                          # cpu variant (default)
#   NPERF_REFS_VARIANT=cuda tools/setup_refs_env.sh  # gpu variant
#   NPERF_REFS_ENV_DIR=/path NPERF_REFS_VARIANT=... tools/setup_refs_env.sh
#
# Then point the runner at it:
#   # cpu
#   NPERF_PYTHON_TORCH="$DIR/bin/python" JAX_PLATFORMS=cpu uv run nperf --quick
#   # gpu (torch + jax both on the GPU; nitrix-jax via the jax-cuda12 env)
#   NPERF_PYTHON_TORCH_JAX_CUDA12="$DIR/bin/python" \
#     NPERF_PYTHON_JAX_CUDA12=/opt/jax_env/bin/python \
#     uv run nperf --platforms jax-cuda12
set -euo pipefail

VARIANT="${NPERF_REFS_VARIANT:-cpu}"
case "$VARIANT" in
  cpu)  DEFAULT_DIR=/output/nperf-refs-env ;;
  cuda) DEFAULT_DIR=/output/nperf-refs-cuda ;;
  *) echo "NPERF_REFS_VARIANT must be 'cpu' or 'cuda' (got '$VARIANT')" >&2
     exit 2 ;;
esac

# Default off-overlay (root overlay is ~1 GB).  Override with NPERF_REFS_ENV_DIR.
TARGET="${NPERF_REFS_ENV_DIR:-$DEFAULT_DIR}"
# Keep uv's cache on the same large volume so big wheels can hardlink in (and
# never fill the root overlay).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/output/uv-cache}"
NITRIX_SRC="${NITRIX_SRC:-$(cd "$(dirname "$0")/../../nitrix" && pwd)}"
JAX_SPEC="${JAX_SPEC:-jax==0.10.0}"

echo "variant   : $VARIANT"
echo "refs env  : $TARGET"
echo "uv cache  : $UV_CACHE_DIR"
echo "nitrix src: $NITRIX_SRC"

if [ -x "$TARGET/bin/python" ] && \
   "$TARGET/bin/python" -c 'import torch, torch_geometric, jax, nitrix' \
     2>/dev/null; then
  echo "refs env already complete (torch+pyg+jax+nitrix import) -- skipping."
  exit 0
fi

uv venv "$TARGET" --python 3.13

if [ "$VARIANT" = cpu ]; then
  # torch+cpu is only on the PyTorch index; jax/numpy/nitrix/torch_geometric on
  # PyPI.  unsafe-best-match lets uv pick the +cpu local-version torch.
  TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1+cpu}"
  echo "torch     : $TORCH_SPEC"
  uv pip install --python "$TARGET/bin/python" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    "$TORCH_SPEC" "$JAX_SPEC" "numpy>=2" "scipy>=1.13" "scikit-learn>=1.4" \
    torch_geometric -e "$NITRIX_SRC"
  PROBE_PLATFORMS=cpu
else
  # CUDA: default-PyPI torch is the cuda build; jax[cuda12] brings its own cuda
  # libs.  Leave torch unpinned (latest compatible cuda wheel) unless overridden.
  TORCH_SPEC="${TORCH_SPEC:-torch}"
  echo "torch     : $TORCH_SPEC (cuda)"
  uv pip install --python "$TARGET/bin/python" \
    "jax[cuda12]==${JAX_SPEC#jax==}" "$TORCH_SPEC" "numpy>=2" \
    "scipy>=1.13" "scikit-learn>=1.4" torch_geometric -e "$NITRIX_SRC"
  PROBE_PLATFORMS=cuda
fi

echo "--- verify ---"
JAX_PLATFORMS="$PROBE_PLATFORMS" "$TARGET/bin/python" - <<'PY'
import torch, torch_geometric, jax, nitrix
print('torch ', torch.__version__, '| cuda', torch.cuda.is_available())
print('pyg   ', torch_geometric.__version__)
print('jax   ', jax.__version__, '| devices', jax.devices())
print('nitrix', getattr(nitrix, '__version__', '?'))
PY
echo "refs env ready: $TARGET/bin/python"
