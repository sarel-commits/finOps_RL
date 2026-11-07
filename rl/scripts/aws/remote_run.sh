#!/bin/bash
# Runs ON the GPU instance (Ubuntu 22.04 DLAMI, 1xH100). Uploaded with the rl/ tree to ~/rl.
#   1. tools: uv, bubblewrap   2. env venv + tests (bwrap backend)   3. vLLM serving Qwen3.8-27B (bf16, pinned rev)
#   4. calibration: N tasks x K samples on the train split + ITBench-Lite eval rollouts
# Usage: bash ~/rl/scripts/aws/remote_run.sh [N_TASKS] [K]
set -euo pipefail
N=${1:-200}; K=${2:-8}
MODEL=Qwen/Qwen3.8-27B
REV=8c0b3d30f3ae96f5c7f774de704bf0ada83e422a
cd ~/rl
exec > >(tee -a ~/run.log) 2>&1
echo "== $(date -u) start N=$N K=$K"

sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap ninja-build >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# --- env venv + tests on Linux (first real test of the bwrap backend) ---
uv venv .venv-vf --python 3.12 -q --allow-existing
VIRTUAL_ENV=.venv-vf uv pip install -q -e ".[verifiers,dev]"
.venv-vf/bin/python - <<'EOF'
from finops_gym.sandbox import resolve_backend, SandboxConfig
print("sandbox backend:", resolve_backend(SandboxConfig().backend))
EOF
.venv-vf/bin/python -m pytest tests -q -k "not prompts_match" 2>&1 | tail -5   # prompt-parity test needs ../scripts (not uploaded)
.venv-vf/bin/vf-validate finops-rca --taskset.split val -n 3 --runtime.type subprocess 2>&1 | tail -5 || echo "vf-validate failed (non-fatal)"

# --- vLLM ---
uv venv ~/.venv-vllm --python 3.12 -q --allow-existing
VIRTUAL_ENV=~/.venv-vllm uv pip install -q vllm ninja
export PATH=$HOME/.venv-vllm/bin:$PATH   # vLLM JIT-compiles kernels at startup and needs ninja on PATH
VLLM_LOG=~/vllm.log
nohup ~/.venv-vllm/bin/vllm serve $MODEL --revision $REV --served-model-name qwen3.8-27b \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.92 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image":0,"video":0}' \
  --host 127.0.0.1 --port 8000 > $VLLM_LOG 2>&1 &
VPID=$!
for i in $(seq 1 180); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $VPID 2>/dev/null || break; sleep 10; done
curl -sf localhost:8000/v1/models >/dev/null || { echo "vLLM did not come up"; tail -50 $VLLM_LOG; exit 1; }
echo "== $(date -u) vLLM up"

# --- calibration ---
PY=.venv-vf/bin/python
$PY scripts/calibrate.py --split train --n-tasks $N --k $K --base-url http://127.0.0.1:8000/v1 --workers 48
sed -i "s#/Users/sarel_arcora/finops-eval/data/itbench-lite/snapshots/finops/v0.1-finops-anomaly#$HOME/itbench#g" data/itbench_lite.jsonl
$PY scripts/calibrate.py --split itbench_lite --n-tasks 15 --k $K --base-url http://127.0.0.1:8000/v1 --workers 48 || true
echo "== $(date -u) done"
touch ~/DONE
