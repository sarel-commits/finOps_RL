#!/bin/bash
# Eval the RL-trained LoRA (step 32, unmerged on the 8-bit base) on ITBench-Lite with the improved harness
# (budget nudges + prompt v2, non-thinking, 15 x 3 seeds): the exact config of the base model's nothink_nudges_v2 run.
# Swaps the local mlx server to base+adapter, then restores the plain base server at the end.
set -u
cd ~/finops-eval && source .venv/bin/activate
ADAPTER=${ADAPTER:-models/adapters/rl-step32}
CONFIG=${CONFIG:-rl32_nothink_nudges_v2}
LOG=logs/server.log
bash scripts/server.sh stop
echo "=== start $(date) model=qwen3.8-27b-mlx-8bit adapter=$ADAPTER ===" >> $LOG
nohup python -m mlx_lm server --model models/qwen3.8-27b-mlx-8bit --adapter-path "$ADAPTER" --host 127.0.0.1 \
  --port 8080 --max-tokens 16384 --prompt-cache-size ${CACHE:-10} --log-level INFO >> $LOG 2>&1 &
SPID=$!
nohup caffeinate -is -w $SPID >/dev/null 2>&1 &
for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:8080/v1/models >/dev/null && break; sleep 3; done
curl -s -m 5 http://127.0.0.1:8080/v1/models >/dev/null || { echo "adapter server failed to start"; exit 1; }
echo "adapter server up (pid $SPID)"
python scripts/run_bench.py --bench itbench_lite --config $CONFIG --seeds 1 2 3 \
  --model-label "qwen3.8-27b-mlx-8bit+$(basename $ADAPTER)" --budget-nudges --prompt-version v2 ${PAIRS:+--only-pairs $PAIRS} \
  && python scripts/score.py itbench_lite $CONFIG > results/itbench_lite/$CONFIG/score_stdout.txt
echo "bench exit $?"
kill $SPID; sleep 5
bash scripts/server.sh start
echo RL_IMPROVED_DONE
