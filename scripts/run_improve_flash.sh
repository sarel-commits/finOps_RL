#!/bin/bash
# Harness-improvement grid on Together Qwen/Qwen3.8-Flash, full ITBench-Lite (DECISIONS #26). Usage: run_improve_flash.sh <config> <bench> [flags...]
C=$1; B=$2; shift 2
cd ~/finops-eval && source .venv/bin/activate && set -a && source .env && set +a
python scripts/run_bench.py --bench $B --config $C --seeds 1 2 3 --model-label together:Qwen/Qwen3.8-Flash \
  --base-url https://api.together.xyz/v1 --model Qwen/Qwen3.8-Flash --api-key-env TOGETHER_API_KEY --stream --retries 6 "$@"
python scripts/score.py $B $C > results/$B/$C/score_stdout.txt
echo FLASH_IMPROVE_DONE $C
