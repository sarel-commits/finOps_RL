#!/bin/bash
# Runs ON a fresh 1xH100 instance (Ubuntu 22.04 Deep Learning Base AMI). Recreates the run-2 training environment:
# ~/rl (uploaded), ~/itbench (uploaded), ~/.venv-unsloth with the versions run 2 logged
# (Unsloth 2026.11.3, Transformers 5.17.0, vLLM 0.33.0, Torch 2.14 cu130), bubblewrap for the sandbox, ~/run_train.sh.
#   bash ~/rl/scripts/aws/setup_train.sh
set -euo pipefail
exec > >(tee -a ~/setup.log) 2>&1
echo "== $(date -u) setup start"
sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq bubblewrap ninja-build >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv ~/.venv-unsloth --python 3.12 -q --allow-existing
VIRTUAL_ENV=~/.venv-unsloth uv pip install -q "unsloth==2026.11.3" "vllm==0.33.0" "transformers==5.17.0" ninja pytest
VIRTUAL_ENV=~/.venv-unsloth uv pip install -q -e ~/rl
~/.venv-unsloth/bin/python -c "import unsloth, vllm, transformers, torch; print('unsloth', unsloth.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
~/.venv-unsloth/bin/python - <<'EOF'
from finops_gym.sandbox import resolve_backend, SandboxConfig
print("sandbox backend:", resolve_backend(SandboxConfig().backend))
EOF
cd ~/rl && ~/.venv-unsloth/bin/python -m pytest tests -q -k "not prompts_match" 2>&1 | tail -3
cat > ~/run_train.sh <<'EOF'
#!/bin/bash
# ~/run_train.sh LOGFILE [train_grpo_unsloth.py args...]
LOG=$1; shift
cd ~/rl && export PATH=$HOME/.venv-unsloth/bin:$PATH
nohup ~/.venv-unsloth/bin/python scripts/train_grpo_unsloth.py "$@" > "$LOG" 2>&1 &
echo "train pid $!"
EOF
chmod +x ~/run_train.sh
echo "== $(date -u) setup done"
touch ~/SETUP_DONE
