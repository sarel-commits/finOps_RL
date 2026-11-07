#!/usr/bin/env python
"""Export eval tasks as Harbor tasks (https://docs.harborframework.com/tasks/), to benchmark any agent/model
(Claude Code, Codex, terminus, a local vLLM model...) on the same tasks with `harbor run`, or through verifiers'
HarborTaskset. Harbor is used for EVAL only; RL training uses the native finops-rca taskset.

  python rl/scripts/export_harbor.py --split itbench_lite --out rl/harbor/itbench_lite
  python rl/scripts/export_harbor.py --split test_synth --n 50 --out rl/harbor/test_synth

Per task (Harbor schema_version 1.4):
  instruction.md          the eval prompt, adapted: answer is written to /app/answer.json (CLI agents have no
                          <answer> channel); the JSON format is unchanged
  task.toml               no network for the agent; verifier in the shared container
  environment/Dockerfile  python:3.12-slim + pandas/numpy, /app holds ONLY anomaly.json + data.csv
  tests/test.sh + verify.py + label.json   uploaded by Harbor to /tests only AFTER the agent finishes
  solution/solve.sh       writes the gold answer (Harbor oracle sanity check: must score 1.0)
Reward: /logs/verifier/reward.json {"reward": 0.5*F1 + 0.5*exact, "exact", "f1", "official_root_cause_resource"},
the same metric definitions as finops_gym/reward.py and scripts/score.py.
"""
import argparse, json, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finops_gym import tasks as T  # noqa: E402
from finops_gym.generator import materialize  # noqa: E402
from finops_gym.prompts import task_prompt  # noqa: E402

FILE_NOTE = ("\n\nYou are working in /app (anomaly.json and data.csv are there). Python 3.12 with pandas and numpy is "
             "installed. When you are done, write your final answer as a single JSON object in exactly the format "
             "above to /app/answer.json (no <answer> tags in the file).")

VERIFY_PY = r'''import json, re, sys
def nn(s): return str(s or "").strip().lower()
def nt(s): return re.sub(r"[\s_\-]+", " ", str(s or "")).strip().lower()
gt = {(nn(x["name"]), nt(x["type"])) for x in json.load(open("/tests/label.json"))}
out = {"reward": 0.0, "exact": 0.0, "f1": 0.0, "official_root_cause_resource": 0.0, "format_ok": 0.0}
try:
    a = json.load(open("/app/answer.json"))
    p = {(nn(r.get("name")), nt(r.get("type"))) for r in a["resource"]}
    tp = len(gt & p); pr = tp / len(p) if p else 0.0; rc = tp / len(gt) if gt else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    out.update(format_ok=1.0, exact=float(p == gt), f1=f1, official_root_cause_resource=float(gt <= p))
    out["reward"] = 0.5 * f1 + 0.5 * out["exact"]
except Exception as e:
    print("unparseable /app/answer.json:", e, file=sys.stderr)
json.dump(out, open("/logs/verifier/reward.json", "w"))
print(out)
'''

TEST_SH = """#!/bin/bash
mkdir -p /logs/verifier
python3 /tests/verify.py
"""

DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3
WORKDIR /app
COPY anomaly.json data.csv /app/
"""


def task_toml(name: str, regime: str, mechanism: str) -> str:
    return f'''schema_version = "1.4"

[task]
name = "finops-rca/{name}"

[metadata]
category = "finops"
regime = "{regime}"
mechanism = "{mechanism}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 3600.0

[environment]
network_mode = "no-network"
workdir = "/app"
cpus = 2
memory_mb = 4096
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="itbench_lite")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    tasks = T.load(T.manifest_path(a.split))[: a.n]
    for t in tasks:
        d = out / t.task_id
        if d.exists():
            shutil.rmtree(d)
        env = d / "environment"
        materialize(t, env)                     # anomaly.json + data.csv only
        (env / "Dockerfile").write_text(DOCKERFILE)
        (d / "instruction.md").write_text(task_prompt(t.anomaly) + FILE_NOTE + "\n")
        (d / "task.toml").write_text(task_toml(t.task_id, t.regime, t.mechanism))
        (d / "tests").mkdir()
        (d / "tests/test.sh").write_text(TEST_SH)
        (d / "tests/verify.py").write_text(VERIFY_PY)
        (d / "tests/label.json").write_text(json.dumps(t.label))
        (d / "solution").mkdir()
        (d / "solution/solve.sh").write_text(
            "#!/bin/bash\ncat > /app/answer.json <<'EOF'\n" + json.dumps({"resource": t.label}) + "\nEOF\n")
        for f in ("tests/test.sh", "solution/solve.sh"):
            (d / f).chmod(0o755)
    print(f"exported {len(tasks)} Harbor tasks to {out}")


if __name__ == "__main__":
    main()
