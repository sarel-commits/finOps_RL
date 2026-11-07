#!/bin/bash
# Re-run the 32 runs the local 8-bit baseline failed on a Together model. Usage: run_api_failed.sh <together_model_id> <config_name>
# Same agent/prompt v1/20 steps/seeds; provider-default sampling with enable_thinking=false. DECISIONS #24.
M=$1; CFG=$2
cd ~/finops-eval && source .venv/bin/activate && set -a && source .env && set +a
COMMON="--model-label together:$M --base-url https://api.together.xyz/v1 --model $M --api-key-env TOGETHER_API_KEY --stream --retries 6 --sampling-profile provider_nothink --seeds 1 2 3"
python scripts/run_bench.py --bench itbench_lite --config $CFG $COMMON --only-pairs $(cat results/itbench_lite/failed_pairs_nothink_native.txt)
python scripts/score.py itbench_lite $CFG > results/itbench_lite/$CFG/score_stdout.txt
python scripts/run_bench.py --bench synthetic --config $CFG $COMMON --only-pairs $(cat results/synthetic/failed_pairs_nothink_native.txt)
python scripts/score.py synthetic $CFG > results/synthetic/$CFG/score_stdout.txt
echo API_DONE $M
