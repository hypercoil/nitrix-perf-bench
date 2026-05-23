#!/usr/bin/env bash
# Build the P2 cross-framework "refs" worker env (torch + jax + nitrix).
#
# Why a separate env: nitrix's own env is jax-only and must NEVER import torch
# (DESIGN §7); the torch reference baseline therefore runs in its *own*
# subprocess under *this* interpreter, selected by the runner via
# `NPERF_PYTHON_TORCH`.  It is a superset env (torch for the baseline, jax +
# nitrix for the shared fp64 oracle every worker rebuilds).
#
# This env is a host artifact, NOT committed.  It lives off the tiny root
# overlay (see TARGET below) because torch is ~1 GB.  Treat the target dir as
# ephemeral (it can vanish across sessions) -- this script is the reproducible
# recipe; re-run it to regenerate.
#
# Usage:
#   tools/setup_refs_env.sh                       # default target + cache
#   NPERF_REFS_ENV_DIR=/path tools/setup_refs_env.sh
#
# Then point the runner at it:
#   NPERF_PYTHON_TORCH="$NPERF_REFS_ENV_DIR/bin/python" \
#     JAX_PLATFORMS=cpu uv run nperf --quick
set -euo pipefail

# Default off-overlay (root overlay is ~1 GB; torch needs far more).  Override
# with NPERF_REFS_ENV_DIR.  /output here is a large scratch volume.
TARGET="${NPERF_REFS_ENV_DIR:-/output/nperf-refs-env}"
# Keep uv's cache on the same large volume so big wheels can hardlink in (and
# never fill the root overlay).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/output/uv-cache}"
NITRIX_SRC="${NITRIX_SRC:-$(cd "$(dirname "$0")/../../nitrix" && pwd)}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.9.1+cpu}"
JAX_SPEC="${JAX_SPEC:-jax==0.10.0}"

echo "refs env  : $TARGET"
echo "uv cache  : $UV_CACHE_DIR"
echo "nitrix src: $NITRIX_SRC"
echo "torch     : $TORCH_SPEC (cpu)"

if [ -x "$TARGET/bin/python" ] && \
   "$TARGET/bin/python" -c 'import torch, jax, nitrix' 2>/dev/null; then
  echo "refs env already complete (torch + jax + nitrix import) -- skipping."
  exit 0
fi

uv venv "$TARGET" --python 3.13

# One resolve so torch+cpu (only on the pytorch index) and jax/numpy/nitrix
# (PyPI) land as a consistent set.  unsafe-best-match lets uv pick the +cpu
# local-version torch from the extra index.
uv pip install --python "$TARGET/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  "$TORCH_SPEC" "$JAX_SPEC" "numpy>=2" -e "$NITRIX_SRC"

echo "--- verify ---"
# Pin cpu for the probe: this is a cpu jaxlib, and some hosts export
# JAX_PLATFORMS=cuda, which would make jax.devices() raise on a cpu build.
JAX_PLATFORMS=cpu "$TARGET/bin/python" - <<'PY'
import torch, jax, nitrix
print('torch ', torch.__version__, '| cuda', torch.cuda.is_available())
print('jax   ', jax.__version__, '| devices', jax.devices())
print('nitrix', getattr(nitrix, '__version__', '?'))
PY
echo "refs env ready: $TARGET/bin/python"
