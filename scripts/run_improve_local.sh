#!/bin/bash
# Harness-improvement chain on local Qwen3.8-27B 8-bit (DECISIONS #26). Checkpointed; run under caffeinate.
cd ~/finops-eval && source .venv/bin/activate
bash scripts/server.sh status | grep -q data || bash scripts/server.sh start
run() { B=$1; C=$2; shift 2; python scripts/run_bench.py --bench $B --config $C --seeds 1 2 3 "$@" && python scripts/score.py $B $C > results/$B/$C/score_stdout.txt; }
run itbench_lite nothink_nudges_v2        --budget-nudges --prompt-version v2
run itbench_lite nothink_nudges_v2_guard  --budget-nudges --prompt-version v2 --repeat-guard
run itbench_lite think_medium_nudges_v2_guard --thinking --reasoning-effort medium --budget-nudges --prompt-version v2 --repeat-guard
run synthetic    nothink_nudges_v2        --budget-nudges --prompt-version v2
echo LOCAL_IMPROVE_DONE
