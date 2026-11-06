#!/bin/bash
# Full benchmark chain (checkpointed; safe to re-run). Wrapped in caffeinate by the caller.
cd ~/finops-eval && source .venv/bin/activate
bash scripts/server.sh status | grep -q data || bash scripts/server.sh start
python scripts/run_bench.py --bench itbench_lite --config nothink_native --seeds 1 2 3
python scripts/score.py itbench_lite nothink_native > results/itbench_lite/nothink_native/score_stdout.txt
python scripts/run_bench.py --bench synthetic --config nothink_native --seeds 1 2 3
python scripts/score.py synthetic nothink_native > results/synthetic/nothink_native/score_stdout.txt
python scripts/run_bench.py --bench itbench_lite --config think_native --thinking --seeds 1 2 3
python scripts/score.py itbench_lite think_native > results/itbench_lite/think_native/score_stdout.txt
echo ALL_RUNS_DONE
