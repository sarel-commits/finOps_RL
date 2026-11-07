#!/bin/bash
# Re-ordered local chain (DECISIONS #28): wait for the in-flight nothink_nudges_v2 run, then thinking first.
cd ~/finops-eval && source .venv/bin/activate
while kill -0 52057 2>/dev/null; do sleep 30; done
run() { B=$1; C=$2; shift 2; python scripts/run_bench.py --bench $B --config $C --seeds 1 2 3 "$@" && python scripts/score.py $B $C > results/$B/$C/score_stdout.txt; }
run itbench_lite nothink_nudges_v2        --budget-nudges --prompt-version v2   # no-op if complete (checkpointed); scores it
run itbench_lite think_medium_nudges_v2_guard --thinking --reasoning-effort medium --budget-nudges --prompt-version v2 --repeat-guard
run synthetic    nothink_nudges_v2        --budget-nudges --prompt-version v2
run itbench_lite nothink_nudges_v2_guard  --budget-nudges --prompt-version v2 --repeat-guard
echo LOCAL_IMPROVE2_DONE
