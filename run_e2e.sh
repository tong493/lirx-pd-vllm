#!/bin/bash
# TAPID prefill-under-vLLM launcher.
#
# TAPID_REPO points at the gpu_daemon repo (branch vllm_qwen3-6-27B_dev) that
# carries python/tapid_vllm (the door) and models/model_assembly/.../vllm_adapter.py
# (the model contract). Both $TAPID_REPO and $TAPID_REPO/python must be on
# PYTHONPATH: the former imports `models.*`, the latter `tapid.*` and
# `tapid_vllm`. TAPID_PY_LIB overrides the libtapid_py.so search if needed.
TAPID_REPO="${TAPID_REPO:-/data/ajhou/repos/vllm-tapid/tapid}"
VLLM_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$VLLM_REPO_DIR:$TAPID_REPO:$TAPID_REPO/python:$PYTHONPATH"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_LOGGING_LEVEL=INFO

PYTHON="$VLLM_REPO_DIR/.venv/bin/python"
[ -x "$PYTHON" ] || PYTHON=python

exec stdbuf -o0 -e0 timeout -s KILL 2400 "$PYTHON" -u "$VLLM_REPO_DIR/run_tapid_vllm.py" "$@"
