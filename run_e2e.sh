#!/bin/bash
# TAPID prefill-under-vLLM launcher. No paths are hardcoded: everything comes
# from the environment or from where this script lives.
#
# Required:
#   TAPID_REPO      gpu_daemon checkout carrying the integration (branch
#                   vllm_integration): python/tapid_vllm + models/.../vllm_adapter.py.
#                   Both $TAPID_REPO and $TAPID_REPO/python are put on PYTHONPATH.
# Optional:
#   TAPID_PY_LIB    path to libtapid_py.so (default: $TAPID_REPO/build/libtapid_py.so)
#   TAPID_PYTHON    python interpreter to use (default: $VLLM_REPO/.venv/bin/python
#                   if present, else python3/python from PATH)
#   TAPID_TIMEOUT   launcher timeout in seconds (default 2400)
#   CUDA_VISIBLE_DEVICES  pick the GPU (the TAPID door uses exactly one)
#
# Example:
#   export TAPID_REPO=$HOME/gpu_daemon
#   export CUDA_VISIBLE_DEVICES=0
#   ./run_e2e.sh --model /path/to/Qwen3.6-27B --max-model-len 2048 --max-tokens 1
set -eu

: "${TAPID_REPO:?Set TAPID_REPO to the gpu_daemon checkout, e.g. export TAPID_REPO=\$PWD/gpu_daemon}"
[ -d "$TAPID_REPO/python/tapid_vllm" ] || {
  echo "ERROR: $TAPID_REPO looks wrong: python/tapid_vllm not found (need the vllm_integration branch)" >&2
  exit 1
}

VLLM_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${TAPID_PYTHON:-}"
if [ -z "$PYTHON" ]; then
  PYTHON="$VLLM_REPO_DIR/.venv/bin/python"
  [ -x "$PYTHON" ] || PYTHON="$(command -v python3 || command -v python)"
fi

export PYTHONPATH="$VLLM_REPO_DIR:$TAPID_REPO:$TAPID_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# Let the door find its lib without extra env: point it at the checkout build.
export TAPID_PY_LIB="${TAPID_PY_LIB:-$TAPID_REPO/build/libtapid_py.so}"

exec stdbuf -o0 -e0 timeout -s KILL "${TAPID_TIMEOUT:-2400}" "$PYTHON" -u "$VLLM_REPO_DIR/run_tapid_vllm.py" "$@"
