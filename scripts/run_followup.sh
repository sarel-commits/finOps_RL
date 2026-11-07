#!/bin/bash
# Follow-up experiment chain (DECISIONS #19). Checkpointed; safe to re-run. Run under caffeinate.
cd ~/finops-eval && source .venv/bin/activate
bash scripts/server.sh status | grep -q data || bash scripts/server.sh start
python scripts/run_bench.py --bench itbench_lite --config nothink_nudges --budget-nudges --seeds 1 2 3
python scripts/score.py itbench_lite nothink_nudges > results/itbench_lite/nothink_nudges/score_stdout.txt
python scripts/run_bench.py --bench itbench_lite --config nothink_nudges_v2 --budget-nudges --prompt-version v2 --seeds 1 2 3
python scripts/score.py itbench_lite nothink_nudges_v2 > results/itbench_lite/nothink_nudges_v2/score_stdout.txt
python scripts/run_bench.py --bench synthetic --config nothink_nudges_v2 --budget-nudges --prompt-version v2 --seeds 1 2 3
python scripts/score.py synthetic nothink_nudges_v2 > results/synthetic/nothink_nudges_v2/score_stdout.txt
echo ALL_FOLLOWUP_DONE
