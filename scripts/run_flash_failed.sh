#!/bin/bash
# Re-run exactly the runs the local 8-bit model failed (nothink_native baseline) on Together Qwen/Qwen3.8-Flash.
# Same agent, prompt v1, no nudges, non-thinking, model-card sampling, same seeds. DECISIONS #23.
cd ~/finops-eval && source .venv/bin/activate && set -a && source .env && set +a
COMMON="--model-label together:Qwen/Qwen3.8-Flash --base-url https://api.together.xyz/v1 --model Qwen/Qwen3.8-Flash --api-key-env TOGETHER_API_KEY --stream --retries 6 --seeds 1 2 3"
python scripts/run_bench.py --bench itbench_lite --config flash_failedruns $COMMON --only-pairs $(cat results/itbench_lite/failed_pairs_nothink_native.txt)
python scripts/score.py itbench_lite flash_failedruns > results/itbench_lite/flash_failedruns/score_stdout.txt
python scripts/run_bench.py --bench synthetic --config flash_failedruns $COMMON --only-pairs $(cat results/synthetic/failed_pairs_nothink_native.txt)
python scripts/score.py synthetic flash_failedruns > results/synthetic/flash_failedruns/score_stdout.txt
echo FLASH_DONE
