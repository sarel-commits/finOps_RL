#!/bin/bash
# Re-run the 32 runs the local 8-bit baseline failed (23 ITBench-Lite + 9 synthetic) on a Claude model via the
# official anthropic SDK. Same agent / prompt v1 / 20 steps / seeds as the GLM/Kimi runs (DECISIONS #24, #31).
# Claude: adaptive thinking (cannot be disabled on Opus 5.5), effort medium, prompt caching, no refusal fallback.
# Usage: run_claude_failed.sh <claude_model_id> <config_name>
M=$1; CFG=$2
cd ~/finops-eval && source .venv/bin/activate && set -a && source .env && set +a
COMMON="--provider anthropic --model $M --model-label anthropic:$M --effort medium --api-key-env ANTHROPIC_API_KEY --retries 3 --seeds 1 2 3"
python scripts/run_bench.py --bench itbench_lite --config $CFG $COMMON --only-pairs $(cat results/itbench_lite/failed_pairs_nothink_native.txt)
python scripts/score.py itbench_lite $CFG > results/itbench_lite/$CFG/score_stdout.txt
python scripts/run_bench.py --bench synthetic --config $CFG $COMMON --only-pairs $(cat results/synthetic/failed_pairs_nothink_native.txt)
python scripts/score.py synthetic $CFG > results/synthetic/$CFG/score_stdout.txt
echo CLAUDE_DONE $M
